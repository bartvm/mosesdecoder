// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>
#include "pymoses/pymoses.h"

namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  boost::thread_specific_ptr<boost::interprocess::managed_shared_memory> segment;
  boost::thread_specific_ptr<boost::interprocess::message_queue> py_to_moses;
  boost::thread_specific_ptr<boost::interprocess::message_queue> moses_to_py;
  boost::thread_specific_ptr<std::unordered_map<std::vector<const Word*>, IntVector> > score_map;
  std::string ThisThreadId(std::string prefix) const;

  const Factor *m_sentenceStart_CSLM, *m_sentenceEnd_CSLM;

public:
  CSLM(const std::string &line);
  ~CSLM();
  void LoadThread();
  void StopThread();

  void IssuePythonRequest(std::vector<const Word*>);
  void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);

  void SendBuffer();
  void SyncBuffer();
  void ClearBuffer();

  virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
};


}
