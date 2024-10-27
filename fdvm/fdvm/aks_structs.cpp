#include "dvm.h"
#include "aks_structs.h"
#include <vector>
#include <map>
#include <string>

using std::vector;
using std::string;
using std::map;

#define DEBUG_LV1 true
#if 1
std::ostream &out = std::cout;
#else
std::ofstream out("_log_debug_info.txt");
#endif

extern SgStatement *dvm_parallel_dir;

SgExpression* findDirect(SgExpression *inExpr, int DIR)
{
    SgExpression *temp = NULL;
    if (inExpr)
    {
        if (inExpr->variant() == DIR)
            return inExpr;
        else
        {
            if (inExpr->lhs())
                temp = findDirect(inExpr->lhs(), DIR);

            if(temp == NULL && inExpr->rhs())
                temp = findDirect(inExpr->rhs(), DIR);
        }
    }
    return temp;
}

static vector<SgSymbol*> fillDataOfArray(SgExpression* on, int& dimInPar)
{
    dimInPar = 0;
    SgExpression* temp = on;
    while (temp)
    {
        dimInPar++;
        temp = temp->rhs();
    }

    vector<SgSymbol*> symbInPar(dimInPar);
    temp = on;
    for (int i = 0; i < dimInPar; ++i)
    {
        symbInPar[i] = temp->lhs()->symbol();
        temp = temp->rhs();
    }
    return symbInPar;
}

static void printError()
{
    err("internal error in across", 424, first_do_par);
    exit(-1);
}

static vector<SageArrayIdxs> GetIdxInParDir(const map<string, SgExpression*>& on, SgExpression *across, bool tie = false)
{
    vector<SageArrayIdxs> ret;

    int dimInPar = 0;
    vector<SgSymbol*> symbInPar;
    vector<SgExpression*> toAnalyze;

    if (across->lhs()->variant() == EXPR_LIST)
        toAnalyze.push_back(across->lhs());
    else
    {
        if (across->lhs()->variant() == DDOT)
            toAnalyze.push_back(across->lhs()->rhs());
        if (across->rhs())
            if (across->rhs()->variant() == DDOT)
                toAnalyze.push_back(across->rhs()->rhs());
    }

    for (int i = 0; i < toAnalyze.size(); ++i)
    {
        across = toAnalyze[i];
        while (across)
        {
            if (symbInPar.size() == 0)
            {
                if (on.size() == 0)
                    printError();
                else if (on.size() == 1)
                    symbInPar = fillDataOfArray(on.begin()->second, dimInPar);
            }

            SgExpression *t = across->lhs();
            int dim = 0;

            if (tie)
            {
                if (t->variant() == ARRAY_REF)
                {
                    if (on.find(t->symbol()->identifier()) == on.end())
                        printError();
                    else
                        symbInPar = fillDataOfArray(on.find(t->symbol()->identifier())->second, dimInPar);
                }
                else if (t->variant() == ARRAY_OP)
                {
                    if (on.find(t->lhs()->symbol()->identifier()) == on.end())
                        printError();
                    else
                        symbInPar = fillDataOfArray(on.find(t->lhs()->symbol()->identifier())->second, dimInPar);
                }
            }

            if (t->variant() == ARRAY_REF)
                t = t->lhs();
            else if (t->variant() == ARRAY_OP)
                t = t->lhs()->lhs();
            else
            {
                if (DEBUG_LV1)
                    out << "!!! unknown variant in ACROSS dir: " << t->variant() << std::endl;
            }

            SgExpression *tmp = t;
            while (tmp)
            {
                dim++;
                tmp = tmp->rhs();
            }

            SageArrayIdxs act;

            act.symb.resize(dim);
            act.dim = dim;
            for (int i = 0; i < dim; ++i)
            {
                act.symb[i].across_left = t->lhs()->lhs()->valueInteger();
                act.symb[i].across_right = t->lhs()->rhs()->valueInteger();
                if (act.symb[i].across_left != 0 || act.symb[i].across_right != 0)
                    act.symb[i].symb = symbInPar[i];
                else if (i < dimInPar)
                    act.symb[i].symb = symbInPar[i];
                else
                    act.symb[i].symb = NULL;
                t = t->rhs();
            }

            ret.push_back(act);
            across = across->rhs();
        }
    }

    return ret;
}

SageAcrossInfo GetLoopsWithParAndAcrDir()
{
    SageAcrossInfo retVal;
    SgStatement *temp = dvm_parallel_dir;

    if (temp->variant() == DVM_PARALLEL_ON_DIR)
    {
        SgExpression *t = findDirect(temp->expr(1), ACROSS_OP);
        SgExpression *tie = findDirect(temp->expr(1), ACC_TIE_OP);

        map<string, SgExpression*> arrays;
        if (t != NULL)
        {
            if (temp->expr(0) && temp->expr(0)->lhs())
            {
                arrays[temp->expr(0)->symbol()->identifier()] = temp->expr(0)->lhs();
                retVal.idxs = GetIdxInParDir(arrays, t);
            }
            else if (tie)
            {
                SgExpression* list = tie->lhs();
                while (list)
                {
                    arrays[list->lhs()->symbol()->identifier()] = list->lhs()->lhs();
                    list = list->rhs();
                }
                retVal.idxs = GetIdxInParDir(arrays, t, true);
            }
            else
                printError();
        }
    }
    return retVal;
}

vector<SageSymbols> GetSymbInParalell(SgExpression *first)
{
    vector<SageSymbols> retval;
    while(first)
    {
        SageSymbols q(first->lhs()->symbol(), -1, 0, 0);
        retval.push_back(q);

        first = first->rhs();
    }
    return retval;
}
