// $Id$
#pragma once

#include <vector>
#include "SingleFactor.h"
#include "moses/Hypothesis.h"
#include <boost/python.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/optional/optional.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp> 

namespace Moses
{
  
  class CSLM : public LanguageModelSingleFactor
  {
    
  // Typedefs of allocators and containers
  typedef boost::interprocess::managed_shared_memory::segment_manager segment_manager_t;
  // This is created because it can be cast to all the other allocators
  typedef boost::interprocess::allocator<void, segment_manager_t> VoidAllocator;
  // Here we create an allocater for the integer vector (the n-grams)
  typedef boost::interprocess::allocator<int, segment_manager_t> IntAllocator;
  typedef boost::interprocess::vector<int, IntAllocator> IntVector;
  
  // The elements of the map take the form of a pair,
  // which needs an allocator as well
  typedef std::pair<const IntVector, float> MapElementType;
  typedef boost::interprocess::allocator<MapElementType, segment_manager_t> MapElementAllocator;
  typedef boost::interprocess::map<IntVector, float, std::less<IntVector>, MapElementAllocator> MapType;
    
  protected:
    int m_state_idx;
    boost::python::object py_run_cslm;
    void IssuePythonRequest(std::vector<const Word*> contextFactor);
//    boost::optional<MapType> requests;
//    boost::interprocess::managed_shared_memory segment;
    std::string thread_id;
    std::string memory_id;
    
  public:
    CSLM(const std::string &line);
    void Load();
    void LoadThread();
    void LoadChild();
    ~CSLM();
    
    virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
    
    void SetFFStateIdx(int state_idx);
    void SyncBuffer();
    void ClearBuffer();
    void RunPython();
    
    // Using a scoped pointer doesn't seem to work, but we can't
    // construct the message queue object because the CSLM class gets
    // constructed before multi-threading starts
    // boost::scoped_ptr<boost::interprocess::message_queue> mq;
    
    virtual void IssueRequestsFor(Hypothesis& hypo, const FFState* input_state);
    
  };
  
  
}
