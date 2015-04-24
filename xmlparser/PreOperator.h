#pragma once
extern "C"{
#include "../third/md5/md5.h"
}
#include "../third/murmur/MurmurHash2_64.h"
#include <windows.h>
#include <string>
#include <fstream>
#include <scarlet/string_help.h>

class PreOperator {
public:

	static std::pair<std::string, std::string > FindCachePath(const std::string & name) {
		auto hash = MurmurHashFile(name.c_str(), MurmurHash64B);
		if (!hash) {
			return{ "", "" };
		}
		auto hash_str = std::to_string(hash);
		char buf[40] = { 0 };
		MD5String(hash_str.c_str(), hash_str.size(), buf, 40);
		auto cache_path = std::string("./cache/") + std::string(buf);
		if (!checkSum(cache_path)) {
			clearCache(cache_path);
			return{ "", buf };
		}
		return{ cache_path, buf };
	}

private:

	static void clearCache(const std::string & name) {
		WIN32_FIND_DATAA wfda;
		HANDLE hFindFile = FindFirstFileA((name + "/*.*").c_str(), &wfda);
		if (hFindFile == INVALID_HANDLE_VALUE) {
			return;
		}
		while (FindNextFileA(hFindFile, &wfda)) {
			if (wfda.cFileName[0] != '.') {
				if (wfda.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) {
					clearCache(name + "/" + wfda.cFileName);
				} else {
					DeleteFileA((name + "/" + wfda.cFileName).c_str());
				}
			}
		}
		CloseHandle(hFindFile);
		RemoveDirectoryA(name.c_str());
	}

	static bool checkSum(const std::string & name) {
		std::ifstream ifs(name + "/index.i");
		if (!ifs) {
			return false;
		}
		std::string filename, md5;
		while (ifs >> filename >> md5) {
			auto hash = MurmurHashFile((name + "/" + filename).c_str(), MurmurHash64B);
			if (!hash) {
				return false;
			}
			auto hash_str = std::to_string(hash);
			char buf[40] = { 0 };
			MD5String(hash_str.c_str(), hash_str.size(), buf, 40);

// 			char md5_buf[40] = { 0 };
// 			if (MD5File((name + "/" + filename).c_str(), md5_buf, 40) == -1) {
// 				return false;
// 			}
			if (buf != md5) {
				return false;
			}
		}
		return true;
	}
};

