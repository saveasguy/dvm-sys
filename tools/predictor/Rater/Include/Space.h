#ifndef SpaceH
#define SpaceH
//////////////////////////////////////////////////////////////////////
//
// Space.h: interface for the Space base class.
//
//////////////////////////////////////////////////////////////////////

#include <vector>
#include <algorithm>
#include <fstream>

class Space {
protected:
	std::vector<long> SizeArray;	// Size of every dimension
	std::vector<long> MultArray;	// Multiplier for each dimension
public:
	Space();
	Space(const std::vector<long>& ASizeArray, std::vector<long> AMultArray);
	Space(const Space &);
 	Space(const std::vector<long>& ASizeArray);
    ~Space();

	long	GetNumInDim(long LI, long dimNum);
	long	GetCenterLI();

	// ������� ���������� �������� ������ ����������, ������� ������ �� shift 
	// �� ��������� dim �� ������� ���������� ��������� �������� �������� 
	long	GetSpecLI(long LI, long dim, int shift); 

	// ��������� ��������� ������ �� ����������� � ������ ������������
	long	GetLI(const std::vector<long> & SI);

	// ����� ������������ ���� ����� ����� ������������
	long	GetDistance(long LI1, long LI2);

	Space&	operator= (const Space &x);

	// ��������� ���������� � ������ ������������ �� ��������� �������
	// (Space Index - SI)
	void	GetSI(long LI, std::vector<long> & SI);

	  // ���������� �������� ������ ������������
	long	GetLSize();

//    inline long GetRank() { return Rank; }

    long GetSize(long AAxis); // ��������� � 1
	long GetMult(long AAxis); // ��������� � 1

	unsigned int Rank() { return SizeArray.size(); }

#ifdef P_DEBUG
	friend std::ostream& operator << (std::ostream& os, const Space& s);
#endif

};

#if defined (__GNUG__) && (__GNUC__  < 3)
inline long abs(long x)
{
	return (x < 0) ? (-x) : (x);
};
#endif

#endif
