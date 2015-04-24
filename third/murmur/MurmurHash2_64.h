#ifndef _MURMURHASH_H
#define _MURMURHASH_H
typedef unsigned __int64 uint64_t;
typedef uint64_t(*LPFNHASH)(const void * key, int len, unsigned int seed);

// 64-bit hash for 64-bit platforms
uint64_t MurmurHash64A(const void * key, int len, unsigned int seed);

// 64-bit hash for 32-bit platforms
uint64_t MurmurHash64B(const void * key, int len, unsigned int seed);

// calc file hash use point hash func
uint64_t MurmurHashFile(const char * filename, LPFNHASH hash_func);

#endif