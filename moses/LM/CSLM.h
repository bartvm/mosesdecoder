// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include "moses/Hypothesis.h"
#include <boost/python.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/optional/optional.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include "pymoses/pymoses.h"
//#include <boost/interprocess/sync/interprocess_mutex.hpp> 

namespace Moses
{
  
  class CSLM : public LanguageModelSingleFactor
  {
    
  protected:
    int m_state_idx;
    void IssuePythonRequest(std::vector<const Word*> contextFactor);
    std::string thread_id;
    std::string memory_id;
    std::string mq_to_id;
    std::string mq_from_id;
    
  public:
    CSLM(const std::string &line);
    void Load();
    void LoadThread();
    ~CSLM();
    
    virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
    
    void SetFFStateIdx(int state_idx);
    void SendBuffer();
    void SyncBuffer();
    void ClearBuffer();
    std::string ThisThreadId(std::string prefix) const;
    
    // boost::scoped_ptr<boost::interprocess::message_queue> mq;
    // Using a scoped pointer doesn't seem to work, but we can't
    // construct the message queue object because the CSLM class gets
    // constructed before multi-threading starts
    
    virtual void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
    
  };
  
  
}
