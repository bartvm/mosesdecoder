// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"

#include <boost/thread.hpp>
#include "Python.h"
#include <numpy/ndarrayobject.h>
#include "util/exception.hh"
#include "moses/Util.h"
#include "moses/StaticData.h"


namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  boost::thread_specific_ptr<bool> loaded;

  // Import Python modules and methods
  PyObject *pGet, *pApplyAsync, *pModule;

  // This mutex prevents simultaneous calls to Python
  boost::mutex mtx_;

  // Thread local storage of batches and results
  // These TSPs must be constructed with this cleanup function
  // This function must be static, else it can't be in the
  // initialization list
  static void PyCleanup(PyObject* pyobj)
  {
    // We're not locking the mutex here, but that should be okay
    // Using this causes a segfault in the end
    // Py_XDECREF(pyobj);
  }
  boost::thread_specific_ptr<PyObject> batch;
  boost::thread_specific_ptr<PyObject> scores;
  boost::thread_specific_ptr<PyObject> async_result;
  boost::thread_specific_ptr<int> batch_count;

  // Add factor IDs for EOS and BOS markers
  const Factor *m_sentenceStart_CSLM, *m_sentenceEnd_CSLM;

  void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
  void IssueRequestFor(std::vector<const Word*>);

public:
  CSLM(const std::string &line);
  ~CSLM();
  void Load();

  void SendBuffer();
  void SyncBuffer();
  void ClearBuffer();

  virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
};


}
