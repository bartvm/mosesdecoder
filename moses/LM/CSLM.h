// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/date_time/posix_time/posix_time.hpp> 
#include "Python.h"
#include <numpy/ndarrayobject.h>
#include "moses/FactorCollection.h"
#include "moses/StaticData.h"
#include "moses/Util.h"
#include "util/exception.hh"

using namespace std;
using namespace boost::interprocess;

namespace Moses {

  class CSLM : public LanguageModelSingleFactor {
    protected:
      // The message queues to communicate with the Python subprocesses
      boost::thread_specific_ptr<message_queue> py2m_tsp;
      boost::thread_specific_ptr<message_queue> m2py_tsp;

      // The NumPy objects that wrap the data
      boost::thread_specific_ptr<mapped_region> ngrams_region_tsp;
      boost::thread_specific_ptr<mapped_region> scores_region_tsp;
      boost::thread_specific_ptr<mapped_region> source_region_tsp;

      // Thread local storage
      boost::thread_specific_ptr<NpyIter> ngrams;
      boost::thread_specific_ptr<NpyIter> scores;
      boost::thread_specific_ptr<NpyIter> source;
      boost::thread_specific_ptr<int> batch_count;
      boost::thread_specific_ptr<int> child_pid;
      // Mutable so that it can be set from Evaluate, which is const
      mutable boost::thread_specific_ptr<InputType> source_sentence;

      // These empty destructors are necessary to avoid double frees
      static void NpyIterCleanup(NpyIter *ptr) {}
      static void InputTypeCleanup(InputType *ptr) {}

      const Factor *m_sentenceStart_CSLM, *m_sentenceEnd_CSLM;
      PyThreadState* state;

      string ThisThreadId(string suffix) const;
      void Cleanup();
      void IssuePythonRequest(vector<const Word*>);

    public:
      CSLM(const string &line);
      ~CSLM();
      void Load();
      void LoadThread();
      void StopThread();

      void SendBuffer();
      void SyncBuffer();
      void ClearBuffer();

      void Evaluate(const Phrase &source,
                    const TargetPhrase &targetPhrase,
                    ScoreComponentCollection &scoreBreakdown,
                    ScoreComponentCollection &estimatedFutureScore) const;
      void Evaluate(const InputType &input,
                    const InputPath &inputPath,
                    const TargetPhrase &targetPhrase,
                    const StackVec *stackVec,
                    ScoreComponentCollection &scoreBreakdown,
                    ScoreComponentCollection *estimatedFutureScore = NULL) const;
      void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
      virtual LMResult GetValue(const vector<const Word*> &contextFactor,
                                State* finalState = 0) const;
  };

}
