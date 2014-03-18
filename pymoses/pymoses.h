#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "scope_aware_allocator.h"
#include "scoped_allocation.h"

// Typedefs of allocators and containers
// Note that we are using a trick taken from the STLdb library; the indexing
// suite requires a default constructor for the allocator, but IPC allocators
// need to be passed a segment manager. We circumvent this by wrapping the
// allocator with a scope-aware wrapper that will use a given segment manager
// to construct the allocator when the default constructor is called
//
// Note that scope_aware_allocator.h and scoped_allocation.h had their headers
// changed, so they can't be copied directly, but they are from here:
// http://sourceforge.net/p/stldb/code/HEAD/tree/branches/r1.2/stldb_lib/stldb/allocators/

typedef boost::interprocess::managed_shared_memory::segment_manager segment_manager_t;
// This is created because it can be cast to all the other allocators
typedef stldb::scope_aware_allocator<boost::interprocess::allocator<void, segment_manager_t> > VoidAllocator;

// Here we create an allocater for the integer vector (the n-grams)
//typedef stldb::scope_aware_allocator<boost::interprocess::allocator<int, segment_manager_t> > IntAllocator;
//typedef boost::interprocess::vector<int, IntAllocator> IntVector;
typedef stldb::scope_aware_allocator<boost::interprocess::allocator<std::string, segment_manager_t> > StringAllocator;
typedef boost::interprocess::vector<std::string, StringAllocator> StringVector;

// The elements of the map take the form of a pair,
// which needs an allocator as well
//typedef std::pair<const IntVector, float> MapElementType;
//typedef stldb::scope_aware_allocator<boost::interprocess::allocator<MapElementType, segment_manager_t> > MapElementAllocator;
//typedef boost::interprocess::map<IntVector, float, std::less<IntVector>, MapElementAllocator> MapType;
typedef std::pair<const StringVector, float> MapElementType;
typedef stldb::scope_aware_allocator<boost::interprocess::allocator<MapElementType, segment_manager_t> > MapElementAllocator;
typedef boost::interprocess::map<StringVector, float, std::less<StringVector>, MapElementAllocator> MapType;
