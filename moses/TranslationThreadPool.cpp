#include "TranslationThreadPool.h"
#include "Timer.h"

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
    VERBOSE(1, "Starting translation thread" << endl);
    Timer thread_timer;
    thread_timer.start();
    ThreadPool::Execute();
    thread_timer.stop();
    VERBOSE(1, "Closing translation thread, took " << thread_timer
               << " seconds" << endl);
    for (iter = ffs.begin(); iter != ffs.end(); ++iter) {
      FeatureFunction *ff = *iter;
      ff->StopThread();
    }
  }

}

#endif
