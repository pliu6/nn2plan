#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"

#if NV_TENSORRT_MAJOR > 1
  #define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
  #define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime
#else
  #define CREATE_INFER_BUILDER createInferBuilder
  #define CREATE_INFER_RUNTIME createInferRuntime
#endif

/**
 * Prefix used for tagging printed log output
 */
#define LOG_GIE "[GIE]  "

/**
 * Logger class for GIE info/warning/errors
 */
class Logger : public nvinfer1::ILogger
{
  void log( Severity severity, const char* msg ) override
  {
    if( severity != Severity::kINFO /*|| enableDebug*/ )
      printf(LOG_GIE "%s\n", msg);
  }
} gLogger;

using namespace std;

bool buildInferEngineFromCaffe(const string& prototxt,          // name for caffe prototxt
                               const string& modelFile,         // name for model
                               const vector<string>& outputs,   // network outputs
                               unsigned int maxBatchSize,       // batch size - NB must be at least as large as the batch we want to run with)
                               ostream& gieModelStream,
                               bool enableDebug = false)
{
  // create API root class - must span the lifetime of the engine usage
  nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();

  builder->setDebugSync(enableDebug);
  builder->setMinFindIterations(3); // allow time for TX1 GPU to spin up
     builder->setAverageFindIterations(2);

  // parse the caffe model to populate the network, then set the outputs
  nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

  bool enableFP16 = builder->platformHasFastFp16();
  printf(LOG_GIE "platform %s FP16 support.\n", enableFP16 ? "has" : "does not have");
  printf(LOG_GIE "loading %s %s\n", prototxt.c_str(), modelFile.c_str());

  nvinfer1::DataType modelDataType = enableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
  const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
    parser->parse(prototxt.c_str(),   // caffe deploy file
            modelFile.c_str(),    // caffe model file
           *network,          // network definition that the parser will populate
            modelDataType);

  if(!blobNameToTensor) {
    printf(LOG_GIE "failed to parse caffe network\n");
    return false;
  }

  // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
  const size_t num_outputs = outputs.size();

  for(size_t n=0; n < num_outputs; n++) {
    nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());

    if( !tensor )
      printf(LOG_GIE "failed to retrieve tensor for output '%s'\n", outputs[n].c_str());
    else
      printf(LOG_GIE "retrieved output tensor '%s'\n", tensor->getName());

    network->markOutput(*tensor);
  }

  // Build the engine
  printf(LOG_GIE "configuring CUDA engine\n");

  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(16 << 20);

  // set up the network for paired-fp16 format
  if(enableFP16) {
    builder->setHalf2Mode(true);
  }

  printf(LOG_GIE "building CUDA engine\n");
  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

  if(!engine) {
    printf(LOG_GIE "failed to build CUDA engine\n");
    return false;
  }

  printf(LOG_GIE "completed building CUDA engine\n");

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
  parser->destroy(); //delete parser;

  // serialize the engine, then close everything down
#if NV_TENSORRT_MAJOR > 1
  nvinfer1::IHostMemory* serMem = engine->serialize();

  if(!serMem) {
    printf(LOG_GIE "failed to serialize CUDA engine\n");
    return false;
  }

  gieModelStream.write((const char*)serMem->data(), serMem->size());
#else
  engine->serialize(gieModelStream);
#endif
  engine->destroy();
  builder->destroy();

  return true;
}

int buildInferEngineFromUff() {

}

void usage() {
  cout << "nn2plan -t caffe <prototxt> <model_file> <max_batch_size> <output1> <output2> ...\nor\n"
          "nn2plan -t uff\n";
}



int main(int argc, char* argv[]) {
  if (argc < 3) {
    usage();
    return 1;
  }

  string mode(argv[2]);
  if (mode == "caffe" && argc > 6) {
    int max_batch_size = 1;
    stringstream batchSizeStream(argv[5]);
    batchSizeStream >> max_batch_size;

    vector<string> outputs;
    for (int i = 6; i < argc; i++) {
      outputs.push_back(argv[i]);
    }
    
    stringstream gieModelStream;
    buildInferEngineFromCaffe(argv[3], argv[4], outputs, max_batch_size, gieModelStream);

    ofstream outFile;
    stringstream planFileName;
    planFileName << argv[4] << "." << max_batch_size << "." << "tensorcache";
    outFile.open(planFileName.str());

    outFile << gieModelStream.rdbuf();
    outFile.close();
    cout << "completed writing serialized engine (plan) to " << planFileName.str() << endl;
  } else if (mode == "uff") {
    // TODO
  } else {
    usage();
    return 1;
  }

  return 0;
}
