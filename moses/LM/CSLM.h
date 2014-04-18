// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include "Python.h"
#include <numpy/ndarrayobject.h>
#include "moses/FactorCollection.h"
#include "moses/StaticData.h"
#include "moses/Util.h"
#include "util/exception.hh"

using namespace std;
using namespace boost::interprocess;

namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  // The message queues to communicate with the Python subprocesses
  boost::thread_specific_ptr<message_queue> py2m_tsp;
  boost::thread_specific_ptr<message_queue> m2py_tsp;

  // The NumPy objects that wrap the data
  boost::thread_specific_ptr<mapped_region> ngrams_region_tsp;
  boost::thread_specific_ptr<mapped_region> scores_region_tsp;
  // static void NpyIterCleanup(NpyIter *ptr) {
  //   // Not sure if we should DECREF
  //   // or just let it be
  // }
  boost::thread_specific_ptr<NpyIter> ngrams;
  boost::thread_specific_ptr<NpyIter> scores;
  boost::thread_specific_ptr<int> batch_count;
  boost::thread_specific_ptr<int> child_pid;

  // A function to retrieve the thread ID
  string ThisThreadId(string suffix) const;

  const Factor *m_sentenceStart_CSLM, *m_sentenceEnd_CSLM;

  PyThreadState* state;

public:
  CSLM(const string &line);
  ~CSLM();
  void LoadThread();
  void StopThread();

  void IssuePythonRequest(vector<const Word*>);
  void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);

  void Load();

  void SendBuffer();
  void SyncBuffer();
  void ClearBuffer();

  virtual LMResult GetValue(const vector<const Word*> &contextFactor, State* finalState = 0) const;
};


}
