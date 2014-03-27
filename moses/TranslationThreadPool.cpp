#include "TranslationThreadPool.h"

#ifdef WITH_THREADS

using namespace std;
using namespace Moses;

namespace Moses
{

  TranslationThreadPool::TranslationThreadPool( size_t numThreads )
    : ThreadPool()
  {
    for (size_t i = 0; i < numThreads; ++i) {
      ThreadPool::m_threads.create_thread(boost::bind(&TranslationThreadPool::Execute,this));
    }
  }

  void TranslationThreadPool::Execute()
  {
    const std::vector<FeatureFunction*> &ffs =
      FeatureFunction::GetFeatureFunctions();
    std::vector<FeatureFunction*>::const_iterator iter;
    for (iter = ffs.begin(); iter != ffs.end(); ++iter) {
      FeatureFunction *ff = *iter;
      ff->LoadThread();
    }
    ThreadPool::Execute();
    for (iter = ffs.begin(); iter != ffs.end(); ++iter) {
      FeatureFunction *ff = *iter;
      ff->StopThread();
    }
  }

}

#endif
