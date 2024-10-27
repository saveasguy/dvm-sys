#ifndef RedGroupH 
#define RedGroupH

//////////////////////////////////////////////////////////////////////
//
// RedGroup.h: interface for the RedGroup class.
//
//////////////////////////////////////////////////////////////////////

#include <vector>

#include "RedVar.h"
#include "DArray.h"

class RedGroup {  
public:
	// ??? ����� �� long CentralProc; // ��������� ������ ��������������� ������ ����������� ������
	VM *vmPtr;
	std::vector<RedVar *> redVars;  // ������ ������������ ����������
	long TotalSize; // ����� ������ � ������ ������������ ��������� ���������� � ������ � �� �������������� ����������
	long CentralProc; // ��������� ������ ��������������� ������ ���������� ������

	RedGroup(VM *AvmPtr);
	virtual ~RedGroup();
	double StartR(DArray * APattern, long ALoopRank, const std::vector<long>& AAxisArray);  // ��������������� // ??? ���������� ����� ��� ����� ���������
	double StartR(AMView * APattern, long ALoopRank, const std::vector<long>& AAxisArray);  // ��������������� // ??? ���������� ����� ��� ����� ���������
	void AddRV(RedVar *ARedVar);
};

#endif 
