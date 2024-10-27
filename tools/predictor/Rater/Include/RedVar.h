#ifndef RedVarH
#define RedVarH

//////////////////////////////////////////////////////////////////////
//
// RedVar.h: interface for the RedVar class.
//
//////////////////////////////////////////////////////////////////////

#include "Vm.h"

class RedVar {
public:
	long RedElmSize;	// ������ ������ �������� ������������ ����������-������� � ������
	long RedArrLength; // ����� ��������� � ������������ ����������-�������
	long LocElmSize; // ������ � ������ ������ �������� ������� � �������������� �����������

	RedVar(long ARedElmSize, long ARedArrLength, long ALocElmSize);
	RedVar();
	virtual ~RedVar();
	long GetSize(); //  ������ � ������ ������������ ����������-������� ������ � �������� �������������� ����������
};

#endif 
