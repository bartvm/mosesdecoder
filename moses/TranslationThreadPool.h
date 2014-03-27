#ifndef moses_TranslationThreadPool_h
#define moses_TranslationThreadPool_h

#include "ThreadPool.h"
#include "FF/FeatureFunction.h"

namespace Moses
{

#ifdef WITH_THREADS

class TranslationThreadPool : public ThreadPool
{
public:
  explicit TranslationThreadPool(size_t numThreads);

private:
  /**
   * The main loop executed by each thread.
   **/
  void Execute();
};

#endif //WITH_THREADS

} // namespace Moses
#endif // moses_TranslationThreadPool_h
