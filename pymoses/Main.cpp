#include <iostream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include "Python.h"
#include <numpy/ndarrayobject.h>

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

void init_numpy() {
  import_array();
}
//
PyObject* LoadPython() {
// void LoadPython() {
  // Load Python
  Py_Initialize();
  init_numpy();
  PyObject* pGet;
  // Load the module and functions
  PyObject* pName = PyString_FromString("cslm");
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  Py_INCREF(pModule);
  if (pModule != NULL) {
    pGet = PyObject_GetAttrString(pModule, "get");
    if (!pGet || !PyCallable_Check(pGet)) {
      VERBOSE(1, "Unable to load CSLM Python method" << endl);
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
    } else {
      VERBOSE(1, "Successfully imported Python module" << endl);
    }
  } else {
    VERBOSE(1, "Unable to load CSLM Python module");
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
  }
  return pGet;
}

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
  PyObject* pGet = LoadPython();
  // LoadPython();

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

  int ngrams_nd = 2;
  npy_intp ngrams_dims[2] = {10000, 7};
  PyObject* ngrams_array = PyArray_SimpleNewFromData(ngrams_nd, ngrams_dims,
                                                     NPY_INT,
                                                     ngrams_region.get_address());
  int scores_nd = 1;
  npy_intp scores_dims[1] = {10000};
  PyObject* scores_array = PyArray_SimpleNewFromData(scores_nd, scores_dims,
                                                     NPY_FLOAT,
                                                     scores_region.get_address());

  VERBOSE(1, "Testing Python function" << endl);
  PyObject *pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, ngrams_array);
  PyTuple_SetItem(pArgs, 1, scores_array);
  PyTuple_SetItem(pArgs, 2, PyInt_FromLong(10));
  PyObject_CallObject(pGet, pArgs);

  // Signal that everything is good to go!
  int message = 0;
  py2m.send(&message, sizeof(int), 0);

  // Listen for messages
  message_queue::size_type recvd_size;
  unsigned int priority;
  while (true) {
    message = 0;
    m2py.receive(&message, sizeof(message), recvd_size, priority);
    if (message > 0) {
      PyTuple_SetItem(pArgs, 2, PyInt_FromLong(message));
      PyObject* result = PyObject_CallObject(pGet, pArgs);
      if (result == Py_True) {
        message = 1;
      } else {
        message = 2;
      }
      py2m.send(&message, sizeof(int), 0);
    } else if (message == -1) {
      // Message -1 means that Moses is quitting (CSLM object destructor)
      message_queue::remove(py2m_id.c_str());
      break;
    } else {
      // Something went wrong
      VERBOSE(1, "PyMoss received strange message, destroying message queue"
                 << endl);
      message_queue::remove(py2m_id.c_str());
      break;
    }
  }
  Py_Finalize();
  VERBOSE(1, "PyMoses terminated" << endl);
}
