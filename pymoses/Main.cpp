#include <iostream>
#include "pymoses.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

/**
 * This was copied from moses/Util.h
 *
 * Outputting debugging/verbose information to stderr.
 * Use TRACE_ENABLE flag to redirect tracing output into oblivion
 * so that you can output your own ad-hoc debugging info.
 * However, if you use stderr directly, please delete calls to it once
 * you finished debugging so that it won't clutter up.
 * Also use TRACE_ENABLE to turn off output of any debugging info
 * when compiling for a gui front-end so that running gui won't generate
 * output on command line
 * */
#ifdef TRACE_ENABLE
#define TRACE_ERR(str) do { std::cerr << str; } while (false)
#else
#define TRACE_ERR(str) do {} while (false)
#endif

/** verbose macros
 * */
#define VERBOSE(level,str) { TRACE_ERR(str); }
#define IFVERBOSE(level)

boost::python::object py_run_cslm;

int main(int argc, char* argv[]) {
  // These are the names to access the message queues and shared memory
  std::string thread_id = argv[1];
  std::string memory_id = "memory" + thread_id;
  std::string mq_to_id = "to" + thread_id;
  std::string mq_from_id = "from" + thread_id;

  // Access the message queues
  boost::interprocess::message_queue py_to_moses(boost::interprocess::open_only,
                                                 mq_to_id.c_str());
  boost::interprocess::message_queue moses_to_py(boost::interprocess::open_only,
                                                 mq_from_id.c_str());

  // Access the shared memory segment
  boost::interprocess::managed_shared_memory segment(boost::interprocess::open_only, memory_id.c_str());
  stldb::scoped_allocation<segment_manager_t> scope(segment.get_segment_manager());
  MapType *requests = segment.find<MapType>("MyMap").first;

  // Start Python
  int message;
  VERBOSE(1, "Starting Python for thread " << thread_id << std::endl);
  Py_Initialize();
  try {
    boost::python::object py_cslm = boost::python::import("cslm");
    py_run_cslm = py_cslm.attr("run_cslm");
  } catch(boost::python::error_already_set const &) {
    // Print the error and signal Moses that something is wrong!
    PyErr_Print();
    message = 1;
    py_to_moses.send(&message, sizeof(int), 1);
    boost::interprocess::message_queue::remove(mq_from_id.c_str());
    exit(1);
  }

  // Expose our data
  boost::python::class_<IntVector>("IntVector")
  .def(boost::python::vector_indexing_suite<IntVector>());
  boost::python::class_<MapType>("MapType")
  .def(boost::python::map_indexing_suite<MapType>());

  // Signal that everything is good to go!
  message = 0;
  py_to_moses.send(&message, sizeof(int), 0);

  // Listen for messages
  boost::interprocess::message_queue::size_type recvd_size;
  unsigned int priority;
  while (true) {
    message = 0;
    moses_to_py.receive(&message, sizeof(message), recvd_size, priority);
    if (message == 1) {
      VERBOSE(1, "Python started scoring..." << std::endl);
      // Message 1 means that a batch is ready to be scored
      try {
        // Run the Python method, read out the scores and save them in shared memory
        // Ideally we would want Python to write to shared memory directly, but
        // I don't know how to do that; using shared pointers just gives trouble
        MapType scores = boost::python::extract<MapType>(py_run_cslm(requests));
        for (MapType::iterator it = requests->begin(); it != requests->end(); ++it) {
          it->second = scores.at(it->first);
        }
      } catch(boost::python::error_already_set const &) {
        PyErr_Print();
      }
      // We signal to Moses that the scoring has been done
      message = 1;
      py_to_moses.send(&message, sizeof(int), 0);
    } else if (message == 2) {
      // Message 2 means that Moses is quitting (CSLM object destructor)
      VERBOSE(1, "Stopping Python, destroying message queue" << std::endl);
      boost::interprocess::message_queue::remove(mq_from_id.c_str());
      exit(0);
    } else {
      // Something went wrong
      VERBOSE(1, "Python received error message, destroying message queue" << std::endl);
      boost::interprocess::message_queue::remove(mq_from_id.c_str());
      exit(1);
    }
  }
}

