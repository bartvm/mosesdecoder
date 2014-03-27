
#include "CSLM.h"
#include "moses/FactorCollection.h"

using namespace std;

namespace Moses
{
CSLM::CSLM(const std::string &line)
  :LanguageModelSingleFactor(line)
{
  ReadParameters();

  FactorCollection &factorCollection = FactorCollection::Instance();

  // needed by parent language model classes. Why didn't they set these themselves?
  m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
  m_sentenceStartWord[m_factorType] = m_sentenceStart;

  m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
  m_sentenceEndWord[m_factorType] = m_sentenceEnd;
}

CSLM::~CSLM()
{
}

std::string CSLM::ThisThreadId(std::string prefix) const {
  // For reasons beyond my comprehension, storing the thread ID in a
  // variable doesn't work; multiple threads end up getting the same ID
  // and the messaging queues/shared memory crash. Requesting the thread ID
  // through a function like this seems to work (note: using a
  // thread_shared_ptr isn't any better and results in double free,
  // corruption, wrong pointers, and more!)
  return prefix +
    boost::lexical_cast<std::string>(boost::this_thread::get_id());
}

void CSLM::StopThread()
{
  // Clean up message queues
  std::cout << "Removing message queues" << std::endl;
  /**
   * TODO: Let child process remove message queues instead
   */
  // int message = 2;
  // moses_to_py->send(&message, sizeof(int), 0);
  boost::interprocess::message_queue::remove(ThisThreadId("from").c_str());
  boost::interprocess::message_queue::remove(ThisThreadId("to").c_str());

  // Clean up shared memory
  std::cout << "Removing shared memory" << std::endl;
  boost::interprocess::shared_memory_object::remove(
    ThisThreadId("memory").c_str()
  );
}

void CSLM::LoadThread()
{
  // Setting up message queues
  // 0 signals READY
  // 1 signals NEXT
  // 2 signals EXIT
  // other ERROR
  std::cout << "Setting up message queues" << std::endl;
  py_to_moses.reset(new boost::interprocess::message_queue(
    boost::interprocess::create_only, ThisThreadId("to").c_str(),
    1, sizeof(int))
  );
  moses_to_py.reset(new boost::interprocess::message_queue(
    boost::interprocess::create_only, ThisThreadId("from").c_str(),
    1, sizeof(int))
  );

  // Setting up the managed shared memory segment
  std::cout << "Setting up shared memory" << std::endl;
  segment.reset(new boost::interprocess::managed_shared_memory(
    boost::interprocess::create_only, ThisThreadId("memory").c_str(),
    419242304
  ));
}

LMResult CSLM::GetValue(const vector<const Word*> &contextFactor, State* finalState) const
{
  LMResult ret;
  ret.score = contextFactor.size();
  ret.unknown = false;

  // use last word as state info
  const Factor *factor;
  size_t hash_value(const Factor &f);
  if (contextFactor.size()) {
    factor = contextFactor.back()->GetFactor(m_factorType);
  } else {
    factor = NULL;
  }

  (*finalState) = (State*) factor;

  return ret;
}

}



