#include <iostream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
// #include "Python.h"
//#include <numpy/ndarrayobject.h>

using namespace boost::interprocess;
using namespace std;

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
#define TRACE_ERR(str) do { cerr << str; } while (false)
#else
#define TRACE_ERR(str) do {} while (false)
#endif

/** verbose macros
 * */
#define VERBOSE(level,str) { TRACE_ERR(str); }
#define IFVERBOSE(level)

void OpenSharedMemory(string name, mapped_region* region) {
  shared_memory_object shm_obj(
    open_only, name.c_str(),
    read_write
  );
  region = new mapped_region(
    shm_obj, read_write
  );
}

void OpenMessageQueue(string name, message_queue* mq) {
  mq = new message_queue(open_only, name.c_str());
}

// void init_numpy() {
//   import_array();
// }
// 
// PyObject* LoadPython() {
//   // Load Python
//   Py_Initialize();
//   init_numpy();
//   PyObject* pGet;
//   // Load the module and functions
//   PyObject* pName = PyString_FromString("cslm");
//   PyObject* pModule = PyImport_Import(pName);
//   Py_DECREF(pName);
//   if (pModule != NULL) {
//     VERBOSE(1, "AAAH");
//     pGet = PyObject_GetAttrString(pModule, "get");
//     if (!pGet || !PyCallable_Check(pGet)) {
//       if (PyErr_Occurred()) {
//         PyErr_Print();
//       }
//       // UTIL_THROW2("Unable to load Python methods apply_async and/or get");
//       VERBOSE(1, "AAAH");
//     } else {
//       VERBOSE(1, "Successfully imported" << endl);
//     }
//   } else {
//     VERBOSE(1, "MUUUH");
//     if (PyErr_Occurred()) {
//       PyErr_Print();
//     }
//     // UTIL_THROW2("Unable to load Python module cslm_pool");
//   }
//
//   return pGet;
// }

int main(int argc, char* argv[]) {
  // These are the names to access the message queues and shared memory
  string thread_id  = argv[1];
  string ngrams_id  = thread_id + "ngrams";
  string scores_id  = thread_id + "scores";
  string m2py_id = thread_id + "m2py";
  string py2m_id = thread_id + "py2m";

  // Start Python
  // int message;
  VERBOSE(1, "Starting Python for thread " << thread_id << endl);
  // PyObject* pGet = LoadPython();

  // Access the message queues
  message_queue m2py(open_only, m2py_id.c_str());
  message_queue py2m(open_only, py2m_id.c_str());

  // // Access the shared memory segment
  shared_memory_object ngrams_shm_obj(open_only, ngrams_id.c_str(), read_write);
  shared_memory_object scores_shm_obj(open_only, scores_id.c_str(), read_write);
  mapped_region ngrams_region(ngrams_shm_obj, read_write);
  mapped_region scores_region(scores_shm_obj, read_write);

  char *mem = static_cast<char*>(ngrams_region.get_address());
  bool success(true);
  for(size_t i = 0; i < ngrams_region.get_size(); ++i) {
    if(*mem++ != 1) {
      VERBOSE(1, "Error! Shared memory segment data seems corrupted" << endl);
      success = false;
    }
  }
  if (success) {
    VERBOSE(1, "Shared memory check completed" << endl);
  }

  // // Create the Python NumPy wrappers and store iterators over them
  // // int ngrams_nd = 2;
  // // npy_intp ngrams_dims[2] = {10000, 7};
  // // VERBOSE(1, "Starting...");
  // // PyObject* ngrams_array = PyArray_SimpleNewFromData(ngrams_nd, ngrams_dims,
  // //                                                    NPY_INT,
  // //                                                    ngrams_address);
  // // VERBOSE(1, "Starting...");
  // // int scores_nd = 1;
  // // npy_intp scores_dims[1] = {10000};
  // // VERBOSE(1, "Starting...");
  // // PyObject* scores_array = PyArray_SimpleNewFromData(scores_nd, scores_dims,
  // //                                                    NPY_FLOAT,
  // //                                                    scores_address);
  // // VERBOSE(1, "Starting...");
  // // PyObject *pArgs = PyTuple_New(2);
  // // PyTuple_SetItem(pArgs, 0, ngrams_array);
  // // PyTuple_SetItem(pArgs, 1, scores_array);

  // // VERBOSE(1, "Starting...");

  // // Signal that everything is good to go!
  int message = 0;
  VERBOSE(1, "Sending OK message..." << endl);
  py2m.send(&message, sizeof(int), 0);
  VERBOSE(1, "Sent OK message" << endl);

  // // Listen for messages
  message_queue::size_type recvd_size;
  unsigned int priority;
  while (true) {
    message = 0;
    m2py.receive(&message, sizeof(message), recvd_size, priority);
    if (message == 1) {
      // PyObject_CallObject(pGet, pArgs);
      // VERBOSE(1, "Running Python" << endl);
      message = 1;
      py2m.send(&message, sizeof(int), 0);
    } else if (message == 2) {
      // Message 2 means that Moses is quitting (CSLM object destructor)
      VERBOSE(1, "Stopping Python, destroying message queue" << endl);
      message_queue::remove(py2m_id.c_str());
      break;
    } else {
      // Something went wrong
      VERBOSE(1, "Python received error message, destroying message queue"
                 << endl);
      message_queue::remove(py2m_id.c_str());
      break;
    }
  }
}

