//-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby

// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment 
// and endian-ness issues if used across multiple platforms.
#include "MurmurHash2_64.h"
#include <windows.h>

uint64_t MurmurHashFile(const char * filename, LPFNHASH hash_func) {
	HANDLE hFile = CreateFileA(filename, FILE_READ_ACCESS, 0, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
	if (hFile == INVALID_HANDLE_VALUE) {
		return 0;
	}
	LARGE_INTEGER file_size = { 0 };
	if (!GetFileSizeEx(hFile, &file_size)) {
		CloseHandle(hFile);
		return 0;
	}
	HANDLE hMapping = CreateFileMappingA(hFile, 0, PAGE_READONLY, 0, 0, nullptr);
	if (hFile == INVALID_HANDLE_VALUE) {
		CloseHandle(hFile);
		return 0;
	}
	char * mapBuffer = (char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
	if (mapBuffer == nullptr) {
		CloseHandle(hMapping);
		CloseHandle(hFile);
		return 0;
	}

	uint64_t h1 = hash_func(mapBuffer, (int)file_size.QuadPart, 0xACF39460);
	uint64_t h2 = hash_func(mapBuffer, (int)file_size.QuadPart, 0xBD180372);

	UnmapViewOfFile(mapBuffer);
	CloseHandle(hMapping);
	CloseHandle(hFile);

	return ((h2 & 0xffffffff00000000) | (h1 & 0xffffffff));
}

uint64_t MurmurHash64A(const void * key, int len, unsigned int seed) {
	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const uint64_t * data = (const uint64_t *)key;
	const uint64_t * end = data + (len / 8);

	while (data != end) {
		uint64_t k = *data++;

		k *= m;
		k ^= k >> r;
		k *= m;

		h ^= k;
		h *= m;
	}

	const unsigned char * data2 = (const unsigned char*)data;

	switch (len & 7) {
	case 7: h ^= uint64_t(data2[6]) << 48;
	case 6: h ^= uint64_t(data2[5]) << 40;
	case 5: h ^= uint64_t(data2[4]) << 32;
	case 4: h ^= uint64_t(data2[3]) << 24;
	case 3: h ^= uint64_t(data2[2]) << 16;
	case 2: h ^= uint64_t(data2[1]) << 8;
	case 1: h ^= uint64_t(data2[0]);
		h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	return h;
}

uint64_t MurmurHash64B(const void * key, int len, unsigned int seed) {
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	unsigned int h1 = seed ^ len;
	unsigned int h2 = 0;

	const unsigned int * data = (const unsigned int *)key;

	while (len >= 8) {
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;

		unsigned int k2 = *data++;
		k2 *= m; k2 ^= k2 >> r; k2 *= m;
		h2 *= m; h2 ^= k2;
		len -= 4;
	}

	if (len >= 4) {
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;
	}

	switch (len) {
	case 3: h2 ^= ((unsigned char*)data)[2] << 16;
	case 2: h2 ^= ((unsigned char*)data)[1] << 8;
	case 1: h2 ^= ((unsigned char*)data)[0];
		h2 *= m;
	};

	h1 ^= h2 >> 18; h1 *= m;
	h2 ^= h1 >> 22; h2 *= m;
	h1 ^= h2 >> 17; h1 *= m;
	h2 ^= h1 >> 19; h2 *= m;

	uint64_t h = h1;

	h = (h << 32) | h2;

	return h;
}
