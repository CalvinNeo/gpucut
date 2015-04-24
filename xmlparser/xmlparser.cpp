// xmlparser.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "PreOperator.h"
#include "Parser.h"
#include <scarlet/string_help.h>
#include <fstream>
#include <vector>
#include <direct.h>

int _tmain(int argc, _TCHAR* argv[]) {
	_mkdir("./cache");
	if (argc != 2){
		return -1;
	}
	scarlet::string_help sh;
	auto filename = sh.ws_to_str(argv[1]);
	auto cache_path = PreOperator::FindCachePath(filename);
	if (!cache_path.first.empty()) {
		printf_s("%s\n", cache_path.first.c_str());
		return 0;
	}
	if (cache_path.second.empty()) {
		return -1;
	}
	Parser p;
	auto errcode = p.parse(filename);
	if (!errcode.second) {
		printf_s("%s\n", errcode.first.c_str());
		return -1;
	}

	std::vector<std::pair<std::string, std::string>> case_checksum;
	auto cases = p.get_test_case();
	auto cpath = std::string("./cache/") + cache_path.second;
	_mkdir(cpath.c_str());

	for (auto const & k : *cases) {
		auto case_file = cpath + "/case" + k._id + ".dat";
		std::ofstream ofs(case_file, std::ios::binary | std::ios::trunc);
		if (!ofs) {
			return -1;
		}
		auto polys_count = (std::uint32_t)k.m_polys_list->size();
		auto circle_count = (std::uint32_t)k.m_cricle_list->size();
		auto line_count = (std::uint32_t)k.m_line_list->size();
		auto polys_v_count = (std::uint32_t)k.m_polys_v_count;
		ofs.write((char*)(&polys_count), sizeof(uint32_t));
		ofs.write((char*)(&circle_count), sizeof(uint32_t));
		ofs.write((char*)(&line_count), sizeof(uint32_t));
		ofs.write((char*)(&polys_v_count), sizeof(uint32_t));

		for (auto const & k2 : *k.m_polys_list) {
			auto polys_s_v_count = (std::uint32_t)k2->size();
			ofs.write((char*)(&polys_s_v_count), sizeof(uint32_t));
		}

		for (auto const & k2 : *k.m_polys_list) {
			for (auto const & k3 : *k2) {
				auto x = (float)k3.first;
				auto y = (float)k3.second;
				//printf_s("polys point:%.0f %.0f\n", x, y);
				ofs.write((char*)(&x), sizeof(float));
				ofs.write((char*)(&y), sizeof(float));
			}
		}

		for (auto const & k2 : *k.m_cricle_list) {
			auto x = (float)std::get<0>(k2);
			auto y = (float)std::get<1>(k2);
			auto r = (float)std::get<2>(k2);
			//printf_s("cricle x,y,r:%.0f %.0f %.0f\n", x, y, r);
			ofs.write((char*)(&x), sizeof(float));
			ofs.write((char*)(&y), sizeof(float));
			ofs.write((char*)(&r), sizeof(float));
		}

		for (auto const & k2 : *k.m_line_list) {
			auto x1 = (float)std::get<0>(k2);
			auto y1 = (float)std::get<1>(k2);
			auto x2 = (float)std::get<2>(k2);
			auto y2 = (float)std::get<3>(k2);
			//printf_s("line x1,y1,x2,y2:%.0f %.0f %.0f %.0f\n", x1, y1, x2, y2);
			ofs.write((char*)(&x1), sizeof(float));
			ofs.write((char*)(&y1), sizeof(float));
			ofs.write((char*)(&x2), sizeof(float));
			ofs.write((char*)(&y2), sizeof(float));
		}
		ofs.close();

		auto hash = MurmurHashFile(case_file.c_str(), MurmurHash64B);
		if (!hash) {
			return -1;
		}
		auto hash_str = std::to_string(hash);
		char md5_buf[40] = { 0 };
		MD5String(hash_str.c_str(), hash_str.size(), md5_buf, 40);

		case_checksum.push_back({ std::string("case") + k._id + ".dat", md5_buf });
	}
	
	std::ofstream ofs(cpath + "/index.i", std::ios::trunc);
	if (!ofs) {
		return -1;
	}
	for (auto const & k : case_checksum) {
		ofs << k.first << " " << k.second << std::endl;
	}
	printf_s("%s\n", cpath.c_str());
	return 0;
}

