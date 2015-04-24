#pragma once
#include <memory>
#include <list>
#include <cinttypes>
#include <windows.h>
#include <scarlet/defer.h>

class Parser {
public:
	using polys = std::list < std::pair<std::uint16_t, std::uint16_t> > ;
	using polys_ptr = std::shared_ptr < polys > ;
	using polys_list = std::list < polys_ptr > ;
	using polys_list_ptr = std::shared_ptr < polys_list > ;

	using line = std::tuple < std::uint16_t, std::uint16_t, std::uint16_t, std::uint16_t > ;
	using line_list = std::list < line > ;
	using line_list_ptr = std::shared_ptr < line_list > ;

	using cricle = std::tuple < std::uint16_t, std::uint16_t, std::uint16_t > ;
	using cricle_list = std::list < cricle > ;
	using cricle_list_ptr = std::shared_ptr < cricle_list > ;

	struct test_case {
		test_case(const std::string & id, const std::string &  desc, const std::string &  type) :
			_id(id), _desc(desc), _type(type) {

		}
		std::string _id, _desc, _type;
		std::uint32_t m_polys_v_count = 0;
		polys_list_ptr m_polys_list = std::make_shared<polys_list>();
		line_list_ptr m_line_list = std::make_shared<line_list>();
		cricle_list_ptr m_cricle_list = std::make_shared<cricle_list>();
	};

	using test_case_list = std::list < test_case > ;
	using test_case_list_ptr = std::shared_ptr < test_case_list > ;
private:
	enum class ParserStatus {
		Ready,
		TestCaseAttr,
		EntityAttr,
		BoundaryAttr,
		TestCaseClose,
		EntityClose,
		BoundaryClose,
		EntityLine,
		EntityCircle,
		Boundary,
		RootClose,
		Failed,
		End
	};

public:
	test_case_list_ptr get_test_case() const {
		return m_test_case_list;
	}

	std::pair<std::string, bool> parse(const std::string & name) {
		scarlet::defer df;
		HANDLE hFile = CreateFileA(name.c_str(), FILE_READ_ACCESS, 0, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
		if (hFile == INVALID_HANDLE_VALUE) {
			return{ "call CreateFile failed", false };
		}
		df.push([&]() {
			CloseHandle(hFile);
		});
		HANDLE hMapping = CreateFileMappingA(hFile, 0, PAGE_READONLY, 0, 0, nullptr);
		if (hFile == INVALID_HANDLE_VALUE) {
			return{ "call CreateFileMapping failed", false };
		}
		df.push([&]() {
			CloseHandle(hMapping);
		});
		char * mapBuffer = (char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
		if (mapBuffer == nullptr) {
			return{ "call MapViewOfFile failed", false };
		}
		df.push([&]() {
			UnmapViewOfFile(mapBuffer);
		});
		m_mapBuffer = mapBuffer;

		auto errcode = skip_xml_header();
		if (!errcode.second) {
			return errcode;
		}
		errcode = skip_xml_test_root();
		if (!errcode.second) {
			return errcode;
		}

		bool end = false;
		ParserStatus machine_status = ParserStatus::Ready;
		while (!end) {
			switch (machine_status) {
			case Parser::ParserStatus::Ready:
				machine_status = parse_xml_ready();
				break;
			case Parser::ParserStatus::TestCaseAttr:
				machine_status = parse_xml_test_case_attr();
				break;
			case Parser::ParserStatus::EntityAttr:
				machine_status = parse_xml_entity_attr();
				break;
			case Parser::ParserStatus::BoundaryAttr:
				machine_status = parse_xml_boundary_attr();
				break;
			case Parser::ParserStatus::TestCaseClose:
				while (m_mapBuffer[m_cur_pos] != '<') {
					++m_cur_pos;
				}
				machine_status = ParserStatus::Ready;
				break;
			case Parser::ParserStatus::EntityClose:
				while (m_mapBuffer[m_cur_pos] != '<') {
					++m_cur_pos;
				}
				machine_status = ParserStatus::Ready;
				break;
			case Parser::ParserStatus::BoundaryClose:
				while (m_mapBuffer[m_cur_pos] != '<') {
					++m_cur_pos;
				}
				machine_status = ParserStatus::Ready;
				break;
			case Parser::ParserStatus::EntityLine:
				machine_status = parse_xml_entity_line();
				break;
			case Parser::ParserStatus::EntityCircle:
				machine_status = parse_xml_entity_circle();
				break;
			case Parser::ParserStatus::Boundary:
				machine_status = parse_xml_boundary();
				break;
			case Parser::ParserStatus::RootClose:
				machine_status = Parser::ParserStatus::End;
				break;
			case Parser::ParserStatus::Failed:
			{
				char errstr[260] = { 0 };
				sprintf_s(errstr, "err character num:%u", m_cur_pos);
				return{ errstr, false };
			}
			break;
			case Parser::ParserStatus::End:
				end = true;
				break;
			default:
				break;
			}
		}
		return{ "", true };
	}

private:

	ParserStatus parse_xml_boundary() {
		auto poly = std::make_shared<polys>();
		while (_strnicmp(m_mapBuffer + m_cur_pos, "<Vertex>", 8) == 0) {
			m_cur_pos += 8;
			auto end_tag = strstr(m_mapBuffer + m_cur_pos, "</Vertex>");
			if (end_tag == nullptr || (end_tag - (m_mapBuffer + m_cur_pos) - 1) > 16) {
				return ParserStatus::Failed;
			}
			char pb[32] = { 0 }, cx[8] = { 0 }, cy[8] = { 0 };
			strncpy_s(pb, m_mapBuffer + m_cur_pos, end_tag - (m_mapBuffer + m_cur_pos));
			auto c = strstr(pb, ",");
			strncpy_s(cx, pb, int(c - pb));
			c += 1;
			while (*c == ' ') {
				++c;
			}
			strcpy_s(cy, c);
			poly->push_back({ atoi(cx), atoi(cy) });
			m_cur_pos = end_tag - m_mapBuffer;
			m_cur_pos += 8;
			while (m_mapBuffer[m_cur_pos] != '<') {
				++m_cur_pos;
			}
		}
		auto iter = --(m_test_case_list->end());
		iter->m_polys_v_count += poly->size();
		iter->m_polys_list->push_back(poly);
		return ParserStatus::Ready;
	}

	ParserStatus parse_xml_entity_circle() {
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<CenterPoint>", 13) != 0) {
			return ParserStatus::Failed;
		}
		m_cur_pos += 13;
		char cpx[8] = { 0 }, cpy[8] = { 0 }, r[8] = { 0 };
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != ',') {
			++m_tmp_pos;
		}
		strncpy_s(cpx, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] == ' ') {
			++m_cur_pos;
		}
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != '<') {
			++m_tmp_pos;
		}
		strncpy_s(cpy, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<Radius>", 8) != 0) {
			return ParserStatus::Failed;
		}
		m_cur_pos += 8;
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != '<') {
			++m_tmp_pos;
		}
		strncpy_s(r, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		auto iter = --(m_test_case_list->end());
		iter->m_cricle_list->push_back(cricle{ atoi(cpx), atoi(cpy), atoi(r) });
		return ParserStatus::Ready;
	}

	ParserStatus parse_xml_entity_line() {
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<StartPoint>", 12) != 0) {
			return ParserStatus::Failed;
		}
		m_cur_pos += 12;
		char spx[8] = { 0 }, spy[8] = { 0 }, epx[8] = { 0 }, epy[8] = { 0 };
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != ',') {
			++m_tmp_pos;
		}
		strncpy_s(spx, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] == ' ') {
			++m_cur_pos;
		}
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != '<') {
			++m_tmp_pos;
		}
		strncpy_s(spy, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<EndPoint>", 10) != 0) {
			return ParserStatus::Failed;
		}
		m_cur_pos += 10;
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != ',') {
			++m_tmp_pos;
		}
		strncpy_s(epx, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] == ' ') {
			++m_cur_pos;
		}
		m_tmp_pos = m_cur_pos;
		while (m_mapBuffer[m_tmp_pos] != '<') {
			++m_tmp_pos;
		}
		strncpy_s(epy, m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
		m_cur_pos = m_tmp_pos + 1;
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		auto iter = --(m_test_case_list->end());
		iter->m_line_list->push_back(line{ atoi(spx), atoi(spy), atoi(epx), atoi(epy) });
		return ParserStatus::Ready;
	}

	ParserStatus parse_xml_boundary_attr() {
		bool skip = false, end = false;
		while (!end) {
			if (skip) {
				if (m_mapBuffer[m_cur_pos] == '\"') {
					skip = false;
				}
				++m_cur_pos;
			} else {
				switch (m_mapBuffer[m_cur_pos]) {
				case '"':
					skip = true;
					++m_cur_pos;
					break;
				case '>':
					end = true;
					break;
				default:
					++m_cur_pos;
					break;
				}
			}
		}
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		return ParserStatus::Boundary;
	}

	ParserStatus parse_xml_entity_attr() {
		std::string type;
		bool skip = false, end = false;
		while (!end) {
			if (skip) {
				if (m_mapBuffer[m_cur_pos] == '\"') {
					skip = false;
				}
				++m_cur_pos;
			} else {
				switch (m_mapBuffer[m_cur_pos]) {
				case 'T':
				case 't':
					m_cur_pos += 6;
					m_tmp_pos = m_cur_pos;
					while (m_mapBuffer[m_tmp_pos] != '\"') {
						++m_tmp_pos;
					}
					type = std::string(m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
					m_cur_pos = m_tmp_pos + 1;
					break;
				case '"':
					skip = true;
					++m_cur_pos;
					break;
				case '>':
					end = true;
					break;
				default:
					++m_cur_pos;
					break;
				}
			}
		}
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		if (type == "Line") {
			return ParserStatus::EntityLine;
		} else if (type == "Circle") {
			return ParserStatus::EntityCircle;
		}
		return ParserStatus::Failed;
	}

	ParserStatus parse_xml_test_case_attr() {
		std::string id, desc, type;
		bool skip = false, end = false;
		while (!end) {
			if (skip) {
				if (m_mapBuffer[m_cur_pos] == '\"') {
					skip = false;
				}
				++m_cur_pos;
			} else {
				switch (m_mapBuffer[m_cur_pos]) {
				case 'I':
				case 'i':
					m_cur_pos += 4;
					m_tmp_pos = m_cur_pos;
					while (m_mapBuffer[m_tmp_pos] != '\"') {
						++m_tmp_pos;
					}
					id = std::string(m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
					m_cur_pos = m_tmp_pos + 1;
					break;
				case 'D':
				case 'd':
					m_cur_pos += 6;
					m_tmp_pos = m_cur_pos;
					while (m_mapBuffer[m_tmp_pos] != '\"') {
						++m_tmp_pos;
					}
					desc = std::string(m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
					m_cur_pos = m_tmp_pos + 1;
					break;
				case 'T':
				case 't':
					m_cur_pos += 6;
					m_tmp_pos = m_cur_pos;
					while (m_mapBuffer[m_tmp_pos] != '\"') {
						++m_tmp_pos;
					}
					type = std::string(m_mapBuffer + m_cur_pos, m_tmp_pos - m_cur_pos);
					m_cur_pos = m_tmp_pos + 1;
					break;
				case '"':
					skip = true;
					++m_cur_pos;
					break;
				case '>':
					end = true;
					break;
				default:
					++m_cur_pos;
					break;
				}
			}
		}
		m_test_case_list->push_back(test_case{ id, desc, type });
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		return ParserStatus::Ready;
	}

	ParserStatus parse_xml_ready() {
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<TestCase ", 10) == 0) {
			m_cur_pos += 10;
			return ParserStatus::TestCaseAttr;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "<Entity ", 8) == 0) {
			m_cur_pos += 8;
			return ParserStatus::EntityAttr;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "<Boundary ", 10) == 0) {
			m_cur_pos += 10;
			return ParserStatus::BoundaryAttr;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "</TestCase>", 11) == 0) {
			m_cur_pos += 11;
			return ParserStatus::TestCaseClose;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "</Entity>", 9) == 0) {
			m_cur_pos += 9;
			return ParserStatus::EntityClose;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "</Boundary>", 11) == 0) {
			m_cur_pos += 11;
			return ParserStatus::BoundaryClose;
		} else if (_strnicmp(m_mapBuffer + m_cur_pos, "</TestRoot>", 11) == 0) {
			m_cur_pos += 10;
			return ParserStatus::RootClose;
		}
		return ParserStatus::Failed;
	}

	std::pair<std::string, bool> skip_xml_header() {
		if ((unsigned char)m_mapBuffer[0] == 0xef &&
			(unsigned char)m_mapBuffer[1] == 0xbb &&
			(unsigned char)m_mapBuffer[2] == 0xbf) {
			m_cur_pos = 3;
		}
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<?xml version=\"1.0\" ", 20) != 0) {
			return{ "xml file header parse error", false };
		}
		m_cur_pos = strlen("<?xml version=\"1.0\" ");
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		return{ "", true };
	}

	std::pair<std::string, bool> skip_xml_test_root() {
		if (_strnicmp(m_mapBuffer + m_cur_pos, "<TestRoot>", 10) != 0) {
			return{ "not find test case root", false };
		}
		m_cur_pos += strlen("<TestRoot>");
		while (m_mapBuffer[m_cur_pos] != '<') {
			++m_cur_pos;
		}
		return{ "", true };
	}

	std::uint16_t atoi(const char * str) {
		auto len = strlen(str);
		if (len > 4 || len == 0) {
			return (std::uint16_t)atol(str);
		}
		switch (len) {
		case 1:
			return str[0] - '0';
			break;
		case 2:
			return (str[0] - '0') * 10 + (str[1] - '0');
			break;
		case 3:
			return (str[0] - '0') * 100 + (str[1] - '0') * 10 + (str[2] - '0');
			break;
		case 4:
			return (str[0] - '0') * 1000 + (str[1] - '0') * 100 + (str[2] - '0') * 10 + (str[3] - '0');
			break;
		}
		return 0;
	}
private:
	char * m_mapBuffer = nullptr;
	std::uint32_t m_cur_pos = 0, m_tmp_pos = 0;

	test_case_list_ptr m_test_case_list = std::make_shared<test_case_list>();
};

