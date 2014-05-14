# include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

void init_numpy() {
  import_array();
}


PyObject* LoadPython() {
  // Load Python
  Py_Initialize();
  init_numpy();

  // Load the module and functions
  PyObject* pGet;
  PyObject* pModule = PyImport_ImportModule("cslm");
  if (pModule != NULL) {
    pGet = PyObject_GetAttrString(pModule, "get");
    if (!pGet || !PyCallable_Check(pGet)) {
      VERBOSE(1, "Unable to load CSLM Python method. ");
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      Py_DECREF(pModule);
      return NULL;
    } else {
      VERBOSE(1, "Successfully imported Python module" << endl);
    }
  } else {
    VERBOSE(1, "Unable to load CSLM Python module. ");
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    return NULL;
  }
  Py_DECREF(pModule);
  Py_DECREF(pGet);
  return pGet;
}


class mq_handle {
  private:
    boost::scoped_ptr<message_queue> m2py;
    boost::scoped_ptr<message_queue> py2m;
    string m2py_id;
    string py2m_id;
  public:
    mq_handle(string thread_id) {
      m2py_id = thread_id + "m2py";
      py2m_id = thread_id + "py2m";
      m2py.reset(new message_queue(open_only, m2py_id.c_str()));
      py2m.reset(new message_queue(open_only, py2m_id.c_str()));
    }
    ~mq_handle() {
      int message(-1);
      py2m->send(&message, sizeof(int), 0);
      VERBOSE(1, "Signalling Moses that PyMoses is exiting." << endl);
    }
    void send(int* message) {
      py2m->send(message, sizeof(int), 0);
    }
    void receive(int* message) {
      message_queue::size_type recvd_size;
      unsigned int priority;
      m2py->receive(message, sizeof(*message), recvd_size, priority);
      if (recvd_size != sizeof(*message)) {
        VERBOSE(1, "PyMoses communication error; wrong message size." << endl);
        *message = -2;
      }
    }
};


int main(int argc, char* argv[]) {
  // These are the names to access the message queues and shared memory
  VERBOSE(1, "Starting PyMoses" << endl);
  bool conditional(false);
  if (argc > 2) {
    // The extra argument means that this is a conditional model
    conditional = true;
  }
  string thread_id  = argv[1];
  string ngrams_id  = thread_id + "ngrams";
  string scores_id  = thread_id + "scores";
  string source_id;
  if (conditional) {
    source_id  = thread_id + "source";
  }

  // Open the message queues
  mq_handle mqs(thread_id);

  // Alive signal
  VERBOSE(1, "PyMoses is sending OK signal" << endl);
  int status = 0;
  mqs.send(&status);

  // Start Python
  VERBOSE(1, "Starting Python for thread " << thread_id << endl);
  PyObject* pGet = LoadPython();
  if (!pGet) {
    VERBOSE(1, "Terminating." << endl);
    Py_Finalize();
    return 1;
  }

  // Access the shared memory segment
  shared_memory_object ngrams_shm_obj(open_only, ngrams_id.c_str(), read_write);
  shared_memory_object scores_shm_obj(open_only, scores_id.c_str(), read_write);
  shared_memory_object* source_shm_obj;
  if (conditional) {
    source_shm_obj = new shared_memory_object(open_only,
                                              source_id.c_str(),
                                              read_write);
  }
  mapped_region ngrams_region(ngrams_shm_obj, read_write);
  mapped_region scores_region(scores_shm_obj, read_write);
  mapped_region source_region;
  if (conditional) {
    mapped_region source_region_swap(*source_shm_obj, read_write);
    source_region.swap(source_region_swap);
  }

  char *mem = static_cast<char*>(ngrams_region.get_address());
  bool success(true);
  for(size_t i = 0; i < ngrams_region.get_size(); ++i) {
    if(*mem++ != 1) {
      success = false;
      break;
    }
  }
  if (success) {
    VERBOSE(1, "Shared memory check completed successfully" << endl);
  } else {
    VERBOSE(1, "Shared memory segment data seems corrupted. Terminating."
               << endl);
    Py_DECREF(pGet);
    Py_Finalize();
    return 1;
  }

  int ngrams_nd = 2;
  npy_intp ngrams_dims[2] = {25000, 7};
  PyObject* ngrams_array = PyArray_SimpleNewFromData(ngrams_nd, ngrams_dims,
                                                     NPY_INT,
                                                     ngrams_region.get_address());
  int scores_nd = 1;
  npy_intp scores_dims[1] = {25000};
  PyObject* scores_array = PyArray_SimpleNewFromData(scores_nd, scores_dims,
                                                     NPY_FLOAT,
                                                     scores_region.get_address());

  PyObject* source_array;
  if (conditional) {
    int source_nd = 1;
    npy_intp source_dims[1] = {250};
    PyObject* source_array = PyArray_SimpleNewFromData(source_nd, source_dims,
                                                       NPY_INT,
                                                       source_region.get_address());
    if (!source_array) {
      VERBOSE(1, "PyMoses was unable to load NumPy arrays from shared memory. "
                 "Terminating." << endl);
      Py_DECREF(pGet);
      Py_Finalize();
      return 1;
    }
  }

  if (!ngrams_array || !scores_array) {
    VERBOSE(1, "PyMoses was unable to load NumPy arrays from shared memory. "
               "Terminating." << endl);
    Py_DECREF(pGet);
    Py_Finalize();
    return 1;
  }
  PyObject *pArgs;
  if (conditional) {
    pArgs = PyTuple_New(4);
    PyTuple_SetItem(pArgs, 3, source_array);
  } else {
    pArgs = PyTuple_New(3);
  }
  PyTuple_SetItem(pArgs, 0, ngrams_array);
  PyTuple_SetItem(pArgs, 1, scores_array);

  // Signal that everything is good to go!
  VERBOSE(1, "PyMoses is sending OK signal" << endl);
  mqs.send(&status);

  // Listen for messages
  while (true) {
    int batch_size = 0;
    mqs.receive(&batch_size);
    if (batch_size > 0) {
      PyObject* py_batch_size = PyInt_FromSize_t(batch_size);
      PyTuple_SetItem(pArgs, 2, py_batch_size);
      PyObject* result = PyObject_CallObject(pGet, pArgs);
      int message;
      if (result == Py_True) {
        message = 1;
        mqs.send(&message);
        Py_DECREF(result);
      } else {
        VERBOSE(1, "PyMoses received strange value from Python (expected True)"
                   "or call failed. Terminating." << endl);
        Py_DECREF(result);
        break;
      }
    } else if (batch_size == 0) {
      // This means we received a source sentence
      //
    } else if (batch_size == -1) {
      // Message -1 means that Moses is quitting (CSLM object destructor)
      // Let Moses know we got the message so that it can destroy the MQs
      VERBOSE(1, "PyMoses received exit message. Terminating." << endl);
      break;
    } else {
      // Something went wrong
      VERBOSE(1, "PyMoses received strange message. Terminating"
                 << endl);
      break;
    }
  }
  Py_DECREF(pGet);
  Py_DECREF(pArgs);
  Py_Finalize();
  return 0;
}
