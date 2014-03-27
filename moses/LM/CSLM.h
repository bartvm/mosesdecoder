// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include <boost/thread/thread.hpp> // Remove this, only for printing Thread ID
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread/tss.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

namespace Moses
{

class CSLM : public LanguageModelSingleFactor
{
protected:
  boost::thread_specific_ptr<boost::interprocess::managed_shared_memory> segment;
  boost::thread_specific_ptr<boost::interprocess::message_queue> py_to_moses;
  boost::thread_specific_ptr<boost::interprocess::message_queue> moses_to_py;
  std::string ThisThreadId(std::string prefix) const;

public:
  CSLM(const std::string &line);
  ~CSLM();
  void LoadThread();
  void StopThread();

  virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
};


}
