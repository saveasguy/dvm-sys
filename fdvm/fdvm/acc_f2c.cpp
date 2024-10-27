#include "dvm.h"
#include "calls.h"

using std::map;
using std::string;
using std::vector;
using std::pair;
using std::set;
using std::stack;
using std::deque;
using std::make_pair;

#define TRACE 0

// for non linear array list
struct PrivateArrayInfo
{
    string name;
    int dimSize;
    vector<SgExpression*> correctExp;
    int typeRed;
    reduction_operation_list *rsl;
};

struct FunctionParam
{
    const char *name;
    int numParam;
    void(*handler) (SgExpression*, SgExpression *&, const char*, int);

    FunctionParam()
    {
        name = NULL;
        numParam = 0;
        handler = NULL;
    }

    FunctionParam(const char *name_, const int numParam_, void(*handler_) (SgExpression*, SgExpression *&, const char*, int))
    {
        name = name_;
        numParam = numParam_;
        handler = handler_;
    }

    void CallHandler(SgExpression *expr, SgExpression *&retExpr)
    {
        handler(expr, retExpr, name, numParam);
    }
};

//global
map <string, vector<vector<SgType*> > > interfaceProcedures;

// extern 
extern SgStatement *first_do_par;
extern SgExpression *private_list;
extern reduction_operation_list *red_struct_list;
extern SgExpression *dvm_array_list;
extern graph_node *node_list;

// extern from acc_f2c_handlers.cpp
extern void __convert_args(SgExpression *, SgExpression *&, SgExpression *&);
extern void __cmplx_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __minmax_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __mod_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __iand_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __ior_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __ieor_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __arc_sincostan_d_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __atan2d_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __sindcosdtand_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __cotan_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __cotand_handler(SgExpression *, SgExpression *&, const char *name, int);
extern void __ishftc_handler(SgExpression *expr, SgExpression *&retExp, const char *name, int);
extern void __merge_bits_handler(SgExpression *expr, SgExpression *&retExp, const char *name, int);
extern void __not_handler(SgExpression *expr, SgExpression *&retExp, const char *name, int);
extern void __poppar_handler(SgExpression *expr, SgExpression *&retExp, const char *name, int);
extern void __modulo_handler(SgExpression *expr, SgExpression *&retExp, const char *name, int);

// local 
static map<string, FunctionParam> handlersOfFunction;
static set<int> supportedVars;
static map<string, SgSymbol*> fTableOfSymbols;
static vector<PrivateArrayInfo> arrayInfo;
static set<long> labels_num;
static map<string, vector<SgLabel*> > labelsExitCycle;
static set<int> unSupportedVars;
static int cond_generator;
static SgStatement* curTranslateStmt;
static map<string, SgSymbol*> autoTfmReplacing;

static map<SgStatement*, vector<SgStatement*> > insertBefore;
static map<SgStatement*, vector<SgStatement*> > insertAfter;

static map<SgStatement*, SgStatement*> replaced;
static int arrayGenNum;
static int SAPFOR_CONV = 0;

#if TRACE
static int lvl_convert_st = 0;
#endif

// functions
void convertExpr(SgExpression*, SgExpression*&);
void createNewFCall(SgExpression *expr, SgExpression *&retExp, const char *name, int nArgs);


#if TRACE
void printfSpaces(int num)
{
    for (int i = 0; i < num; ++i)
        printf(" ");
}
#endif

static void saveInsertBeforeAfter(map<SgStatement*, vector<SgStatement*> > &after, map<SgStatement*, vector<SgStatement*> > &before)
{
    if (!options.isOn(AUTO_TFM))
        return;
    
    before = insertBefore;
    insertBefore.clear();

    after = insertAfter;
    insertAfter.clear();    
}

static void restoreInsertBeforeAfter(map<SgStatement*, vector<SgStatement*> >& after, map<SgStatement*, vector<SgStatement*> >& before)
{
    if (!options.isOn(AUTO_TFM))
        return;
    
    insertBefore = before;
    insertAfter = after;
}

static void copyToStack(stack<SgStatement*> &newBody, const map<SgStatement*, vector<SgStatement*> > &cont)
{
    if (!options.isOn(AUTO_TFM))
        return;

    if (cont.size())
        for (map<SgStatement*, vector<SgStatement*> >::const_iterator itI = cont.begin(); itI != cont.end(); itI++)
            for (int z = 0; z < itI->second.size(); ++z)
                newBody.push(itI->second[z]);
}

static bool isInPrivate(const string& arr)
{
    for (int z = 0; z < arrayInfo.size(); ++z)
    {
        if (arrayInfo[z].name == arr)
            return true;
    }
    return false;
}

static char* getNestCond()
{
    char buf[32];
    buf[0] = '\0';
    sprintf(buf, "%d", cond_generator);
    cond_generator++;
    char *str = new char[strlen("cond_") + strlen(buf) + 2];
    str[0] = '\0';
    strcat(str, "cond_");
    strcat(str, buf);
    return str;
}

static char* getNewCycleVar(const char *oldVar)
{
    char *str = new char[strlen(oldVar) + 3];
    str[0] = '\0';
    strcat(str, "__");
    strcat(str, oldVar);
    return str;
}

static bool inNewVars(const char *name)
{
    bool ret = false;
    for (size_t i = 0; i < newVars.size(); ++i)
    {
        if (strcmp(name, newVars[i]->identifier()) == 0)
        {
            ret = true;
            break;
        }
    }
    return ret;
}

static void addInListIfNeed(SgSymbol *tmp, int type, reduction_operation_list *tmpR)
{
    stack<SgExpression*> allArraySub;
    stack<pair<SgExpression*, SgExpression*> > allArraySubConv;
    if (tmp)
    {
        if (isSgArrayType(tmp->type()))
        {
            if (isSgArrayType(tmp->type())->dimension() > 0)
            {
                SgExpression *dimList = isSgArrayType(tmp->type())->getDimList();
                PrivateArrayInfo t;
                t.dimSize = isSgArrayType(tmp->type())->dimension();

                int rank = 0;
                while (dimList)
                {
                    allArraySub.push(dimList->lhs());
                    allArraySubConv.push(make_pair(LowerShiftForArrays(tmp, rank, type), UpperShiftForArrays(tmp, rank)));
                    ++rank;
                    dimList = dimList->rhs();
                }

                dimList = isSgArrayType(tmp->type())->getDimList();
                rank = 0;

                while (dimList)
                {
                    SgExpression *ex = allArraySub.top();
                    bool ddot = false;
                    if (ex->variant() == DDOT && ex->lhs() || IS_ALLOCATABLE(tmp))
                        ddot = true;
                    t.correctExp.push_back(LowerShiftForArrays(tmp, rank, type));

                    // swap array's dimentionss
                    if (inNewVars(tmp->identifier()))
                    {
                        if (ddot)
                            dimList->setLhs(*allArraySubConv.top().second - *allArraySubConv.top().first + *new SgValueExp(1));
                        else
                            dimList->setLhs(allArraySubConv.top().first);
                    }

                    allArraySub.pop();
                    allArraySubConv.pop();
                    ++rank;
                    dimList = dimList->rhs();
                }
                t.name = tmp->identifier();
                // 0 for private, 1 for loc and redudction variables
                t.typeRed = type;
                t.rsl = tmpR;
                arrayInfo.push_back(t);
            }
        }
    }
}

static void addRandStateIfNeeded(const string& name)
{
    SgExpression* list = private_list;
    while (list)
    {
        if (list->lhs()->symbol()->identifier() == name)
            return;
        list = list->rhs();
    }

    SgSymbol* uint4_t = new SgSymbol(TYPE_NAME, "uint4", *(current_file->firstStatement()));

    SgFieldSymb* sx = new SgFieldSymb("x", *SgTypeInt(), *uint4_t);
    SgFieldSymb* sy = new SgFieldSymb("y", *SgTypeInt(), *uint4_t);
    SgFieldSymb* sz = new SgFieldSymb("z", *SgTypeInt(), *uint4_t);
    SgFieldSymb* sw = new SgFieldSymb("w", *SgTypeInt(), *uint4_t);

    SYMB_NEXT_FIELD(sx->thesymb) = sy->thesymb;
    SYMB_NEXT_FIELD(sy->thesymb) = sz->thesymb;
    SYMB_NEXT_FIELD(sz->thesymb) = sw->thesymb;
    SYMB_NEXT_FIELD(sw->thesymb) = NULL;

    SgType* tstr = new SgType(T_STRUCT);
    TYPE_COLL_FIRST_FIELD(tstr->thetype) = sx->thesymb;
    uint4_t->setType(tstr);

    SgType* td = new SgType(T_DERIVED_TYPE);
    TYPE_SYMB_DERIVE(td->thetype) = uint4_t->thesymb;
    TYPE_SYMB(td->thetype) = uint4_t->thesymb;

    newVars.push_back(new SgSymbol(VARIABLE_NAME, name.c_str(), td, mod_gpu));
    SgExprListExp* e = new SgExprListExp(*new SgVarRefExp(newVars.back()));
    e->setRhs(private_list);
    private_list = e;
}

void swapDimentionsInprivateList()
{
    SgExpression *tmp = private_list;
    arrayInfo.clear();

    while (tmp)
    {
        addInListIfNeed(tmp->lhs()->symbol(), 0, NULL);
        tmp = tmp->rhs();
    }

    reduction_operation_list *tmpR = red_struct_list;
    while (tmpR)
    {
        SgSymbol *tmp = NULL;
        tmp = tmpR->locvar;
        addInListIfNeed(tmp, 1, tmpR);

        tmp = tmpR->redvar;
        addInListIfNeed(tmp, 1, tmpR);

        tmpR = tmpR->next;
    }
}

//return 'true' if simple operator, 'false' - complex operator
static bool checkLastNode(int var)
{
    bool ret = true;
    if (var == FOR_NODE)
        ret = false;
    else if (var == WHILE_NODE)
        ret = false;
    else if (var == SWITCH_NODE)
        ret = false;
    /*else if (var == LOGIF_NODE)
        ret = false;
        else if (var == ARITHIF_NODE)
        ret = false;*/
    else if (var == IF_NODE)
        ret = false;

    return ret;
}

static void setControlLexNext(SgStatement* &currentSt)
{
    SgStatement *tmp = currentSt;
    if (tmp->variant() == IF_NODE)
    {
        SgStatement *last = tmp->lastNodeOfStmt();
        if (((SgIfStmt*)tmp)->falseBody())
        {
            last = ((SgIfStmt*)tmp)->falseBody();
            for (;;)
            {
                if (last->variant() == ELSEIF_NODE)
                {
                    if (((SgIfStmt*)last)->falseBody())
                        last = ((SgIfStmt*)last)->falseBody();
                    else
                    {
                        last = last->lastNodeOfStmt();
                        break;
                    }
                }
                else
                {
                    last = last->controlParent()->lastNodeOfStmt();
                    break;
                }
            }
        }
        else
            last = tmp->lastNodeOfStmt();

        currentSt = last->lexNext();
    }
    else if (tmp->variant() == FOR_NODE || tmp->variant() == WHILE_NODE || tmp->variant() == SWITCH_NODE)
    {
        if (checkLastNode(currentSt->lastNodeOfStmt()->variant()) == false)
        {
            currentSt = currentSt->lastNodeOfStmt();
            setControlLexNext(currentSt);
        }
        else
            currentSt = currentSt->lastNodeOfStmt()->lexNext();
    }
    else if (tmp->variant() == LOGIF_NODE || tmp->variant() == ARITHIF_NODE)
        currentSt = ((SgIfStmt*)tmp)->lastNodeOfStmt()->lexNext();
    else
    {
        //if (tmp->variant() != ASSIGN_STAT && tmp->variant() != CONT_STAT && tmp->variant() != GOTO_NODE)
        //    printf("  [WARNING: acc_f2c.cpp, line %d] lexNext of %s variant.\n", __LINE__, tag[tmp->variant()]);
        currentSt = currentSt->lexNext();
    }
}

// create lables for EXIT and CYCLE statemets
static void createNewLabel(vector<SgStatement*> &labSt, vector<SgLabel*> &lab, const char *name)
{
    char *str_cont = new char[64];
    str_cont[0] = '\0';
    strcat(str_cont, "label_cycle_");
    strcat(str_cont, name);
    
    if (labelsExitCycle.find(str_cont) != labelsExitCycle.end())
        lab = labelsExitCycle[str_cont];
    else
    {
        SgLabel *lab_cont = GetLabel();
        SgSymbol *symb_cont = new SgSymbol(LABEL_NAME, str_cont);
        LABEL_SYMB(lab_cont->thelabel) = symb_cont->thesymb;

        char *str_exit = new char[64];
        str_exit[0] = '\0';
        strcat(str_exit, "label_exit_");
        strcat(str_exit, name);

        SgLabel *lab_exit = GetLabel();
        SgSymbol *symb_exit = new SgSymbol(LABEL_NAME, str_exit);
        LABEL_SYMB(lab_exit->thelabel) = symb_exit->thesymb;

        lab.push_back(lab_cont);
        lab.push_back(lab_exit);

        labelsExitCycle[string(str_cont)] = lab;
    }
    SgStatement *cycleSt = new SgStatement(LABEL_STAT);
    BIF_LABEL_USE(cycleSt->thebif) = lab[0]->thelabel;

    SgStatement *exitSt = new SgStatement(LABEL_STAT);
    BIF_LABEL_USE(exitSt->thebif) = lab[1]->thelabel;

    labSt.push_back(cycleSt);
    labSt.push_back(exitSt);
}

static void createNewLabel(SgStatement* &labSt, SgLabel *lab)
{
    SgSymbol *symb;
    int labDigit = (int)(lab->thelabel->stateno);

    char *str = new char[32];
    char *digit = new char[32];
    str[0] = digit[0] = '\0';
    strcat(str, "label_");
    sprintf(digit, "%d", labDigit);
    strcat(str, digit);

    symb = new SgSymbol(LABEL_NAME, str);
    LABEL_SYMB(lab->thelabel) = symb->thesymb;
    labSt = new SgStatement(LABEL_STAT);
    BIF_LABEL_USE(labSt->thebif) = lab->thelabel;
}

static void convertLabel(SgStatement *st, SgStatement * &ins, bool ret)
{
    SgLabel *lab = st->label();
    SgStatement *labSt = NULL;
    createNewLabel(labSt, lab);

    if (ret)
        ins = labSt;
    else
        st->insertStmtBefore(*labSt, *st->controlParent());
}

SgStatement* getInterfaceForCall(SgSymbol* s)
{
    SgStatement* searchStmt = cur_func->lexNext();
    SgStatement* tmp;
    string funcName = string(s->identifier());
    enum {SEARCH_INTERFACE,CHECK_INTERFACE, FIND_NAME, SEARCH_INTERNAL,SEARCH_CONTAINS,UNSUCCESS};
    int mode = SEARCH_CONTAINS;
    
    //search internal function
    while(searchStmt&& mode!=UNSUCCESS)
    {
        switch(mode)
        {
        case SEARCH_CONTAINS:
            if(searchStmt->variant() == CONTAINS_STMT)
                mode = SEARCH_INTERNAL;
            searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            break;
        case SEARCH_INTERNAL:
            if(searchStmt->variant() == CONTROL_END)
                mode = UNSUCCESS;
            else if(string(searchStmt->symbol()->identifier()) == funcName)
                return searchStmt;
            else
              searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            break;         
        }
    }
    searchStmt = cur_func->lexNext();
    mode = SEARCH_INTERFACE;
    //search interface in declare section
    while(searchStmt && !isSgExecutableStatement(searchStmt) )
    {
        switch(mode)
        {
        case SEARCH_INTERFACE:
            if(searchStmt->variant() != INTERFACE_STMT)
                searchStmt = searchStmt->lexNext();
            else
                mode = CHECK_INTERFACE;
            break;
        case CHECK_INTERFACE:
            if(searchStmt->symbol()&& string(searchStmt->symbol()->identifier()) != funcName)
            {
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
                mode = SEARCH_INTERFACE;
            }
            else
            {
                mode = FIND_NAME;
                searchStmt = searchStmt->lexNext();
            }
            break;
        case FIND_NAME:
            if(searchStmt->variant() == FUNC_HEDR || searchStmt->variant() == PROC_HEDR)
            {
                if(string(searchStmt->symbol()->identifier()) == funcName)
                    return searchStmt;
                else
                    searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            }
            else if(searchStmt->variant() == MODULE_PROC_STMT)
            {
                searchStmt = searchStmt->lastNodeOfStmt()->lexNext();
            }
            else if(searchStmt->variant() == CONTROL_END)
            {
                mode = SEARCH_INTERFACE;    
                searchStmt = searchStmt->lexNext();
            }
            break;
        }
    }
    return NULL;
}

//TODO: to be removed ??!!

//SgExpression* makePresentExpr(string argName, SgStatement* header)
//{
//    int i = 0;
//    while(header&&(header->variant() != FUNC_HEDR && header->variant()!=PROC_HEDR))
//        header = header->controlParent();
//    if(!header)
//    {
//        printf("  [EXPR ERROR: %s, line %d, user line %d] use PRESENT outside prcodedure or function  \"%s\"\n", __FILE__, __LINE__, first_do_par->lineNumber(), "****");
//        return NULL;
//    }
//    SgExpression* args = header->expr(0)->lhs();
//    while(args)
//        if(string(args->lhs()->symbol()->identifier()) == argName)
//        {
//            SgExpression* presentExpr = &(*(new SgVarRefExp(header->expr(0)->lhs()->lhs()->symbol()) ) & *new SgExprListExp( *new SgValueExp(1) << *(new SgValueExp(i-1))));
//            return presentExpr;
//        }
//        else
//        {
//            args = args->rhs();
//            i++;
//        }
//    return NULL;
//
//}

SgExpression* switchArgumentsByKeyword(const string& name, SgExpression* funcCall, SgStatement* funcInterface)
{
    //get list of arguments names
    vector<string> listArgsNames;
    SgFunctionSymb* s = (SgFunctionSymb*)funcInterface->symbol();
    vector<SgExpression*> resultExprCall(s->numberOfParameters(), (SgExpression*)NULL);
    int useKeywords = false;
    int useOptional = false;
    int useArray = false;

    for (int i = 0; i < s->numberOfParameters(); ++i)
    {
        listArgsNames.push_back(s->parameter(i)->identifier());
        if (s->parameter(i)->attributes() & OPTIONAL_BIT)
            useOptional = true;
    }

    SgExpression* parseExpr;
    if (funcCall->variant() == FUNC_CALL)
        parseExpr = funcCall->lhs();
    else
        parseExpr = funcCall;

    int curArgumentPos = 0;
    while (parseExpr)
    {
        if (parseExpr->lhs()->variant() == KEYWORD_ARG)
        {
            useKeywords = true;
            int newPos = 0;
            string keyword = string(((SgKeywordValExp*)parseExpr->lhs()->lhs())->value());
            while (listArgsNames[newPos] != keyword)
                newPos++;

            resultExprCall[newPos] = parseExpr->lhs()->rhs();
        }
        else if (useKeywords)
            Error("Position argument after keyword", "", 650, first_do_par);
        else
            resultExprCall[curArgumentPos] = parseExpr->lhs();
        curArgumentPos++;
        parseExpr = parseExpr->rhs();
    }

    //check assumed form array
    for (int i = 0; i < resultExprCall.size(); ++i)
    {
        SgSymbol* sarg = s->parameter(i);
        if (isSgArrayType(sarg->type()))
        {
            int needChanged = true;
            SgArrayType* arrT = (SgArrayType*)sarg->type();
            int dims = arrT->dimension();
            SgExpression* dimList = arrT->getDimList();
            while (dimList)
            {
                if (dimList->lhs()->variant() != DDOT)
                {
                    needChanged = false;
                    break;
                }
                else if (dimList->lhs()->rhs())
                {
                    needChanged = false;
                    break;
                }
                dimList = dimList->rhs();
            }

            if (needChanged)
            {
                useArray = true;

                SgArrayType* argType = (SgArrayType*)resultExprCall[i]->symbol()->type();
                SgExprListExp* argInfo = (SgExprListExp*)argType->getDimList();
                SgExpression* tmp;
                int argDims = argType->dimension();

                //TODO: 
                if (argDims != dims)
                {
                    char buf[256];
                    sprintf(buf, "Dimention of the %d formal and actual parameters of '%s' call is not equal", i, name.c_str());
                    Error(buf, "", 651, first_do_par);
                }

                SgExpression* argList = NULL;
                for (int j = 6; j >= 0; --j)
                {
                    if (argInfo->elem(j) == NULL)
                        continue;

                    //TODO: not checked!!                    
                    SgExpression* val = Calculate(&(*UpperBound(resultExprCall[i]->symbol(), j) - *LowerBound(resultExprCall[i]->symbol(), j) + *LowerBound(s->parameter(i), j)));
                    if (val != NULL)
                        tmp = new SgExprListExp(*val);
                    else
                        tmp = new SgExprListExp(*new SgValueExp(int(0)));

                    tmp->setRhs(argList);
                    argList = tmp;
                    val = LowerBound(s->parameter(i), j);
                    if (val != NULL)
                        tmp = new SgExprListExp(*val);
                    else
                        tmp = new SgExprListExp(*new SgValueExp(int(0)));
                    tmp->setRhs(argList);
                    argList = tmp;
                }

                SgArrayRefExp* arrRef = new SgArrayRefExp(*resultExprCall[i]->symbol());
                for (int j = 0; j < dims; ++j)
                    arrRef->addSubscript(*new SgValueExp(0));

                tmp = new SgExprListExp(SgAddrOp(*arrRef));
                tmp->setRhs(argList);
                argList = tmp;
                SgSymbol* aa = s->parameter(i);

                SgTypeRefExp* typeExpr = new SgTypeRefExp(*C_Type(s->parameter(i)->type()));
                resultExprCall[i] = new SgFunctionCallExp(*((new SgDerivedTemplateType(typeExpr, new SgSymbol(TYPE_NAME, "s_array")))->typeName()), *argList);
                resultExprCall[i]->setRhs(typeExpr);
            }
        }
    }

    //change position in call expression if argument passed by keyword
    if (useKeywords || useOptional || useArray)
    {
        int mask = 0;
        SgExpression* maskExpr = new SgValueExp(int(0));
        int bit = 1;
        //change arg -> point to arg when arg is optional
        for (int i = 0; i < resultExprCall.size() - 1; ++i)
        {
            SgSymbol* tmps = s->parameter(i);

            //TODO: WTF ???!
            if ((s->parameter(i)->attributes() & OPTIONAL_BIT) && resultExprCall[i] != NULL)
            {
                /*if(resultExprCall[i]->variant() == VAR_REF && resultExprCall[i]->symbol()->attributes()&OPTIONAL_BIT )
                {
                    SgFunctionSymb* fName = ((SgFunctionSymb *)resultExprCall[i]->symbol()->scope()->symbol());
                    int pos = 0;
                    for(int j = 0; j < fName->numberOfParameters(); ++j)
                        if(string(fName->parameter(j)->identifier()) == string(resultExprCall[j]->symbol()->identifier()))
                        {
                            pos = j;
                            break;
                        }
                        maskExpr = &(*maskExpr | (((*new SgVarRefExp(fName->parameter(0)) >> (*new SgValueExp(pos))) & *new SgValueExp(1)) << *new SgValueExp(i)));
                }
                else*/
                // maskExpr = Calculate(&(*maskExpr | *new SgValueExp(int(1<<i))));
            }
            else if ((s->parameter(i)->attributes() & OPTIONAL_BIT) && resultExprCall[i] == NULL)
            {
                SgTypeRefExp* typeExpr = new SgTypeRefExp(*C_Type(s->parameter(i)->type()));
                resultExprCall[i] = new SgFunctionCallExp(*((new SgDerivedTemplateType(typeExpr, new SgSymbol(TYPE_NAME, "optArg")))->typeName()));
                resultExprCall[i]->setRhs(new SgExprListExp(*typeExpr));
            }
        }

        SgExprListExp* expr = new SgExprListExp();
        SgExprListExp* tmp = expr;
        SgExprListExp* tmp2;
        //insert info-argument at first position

        //insert rguments
        for (int i = 0; i < resultExprCall.size() - 1; ++i)
        {
            tmp->setLhs(resultExprCall[i]);
            tmp->setRhs(new SgExprListExp());
            tmp = (SgExprListExp*)tmp->rhs();
        }

        tmp->setLhs(resultExprCall[resultExprCall.size() - 1]);
        if (funcCall->variant() == FUNC_CALL)
            funcCall->setLhs(expr);
        else
            funcCall = expr;
    }
    return funcCall;
}

SgSymbol* createNewFunctionSymbol(const char *name)
{
    SgSymbol *symb = NULL;
    if (name == NULL)
        name = "__dvmh_tmp_symb";

    if (fTableOfSymbols.find(name) == fTableOfSymbols.end())
    {
        symb = new SgSymbol(FUNCTION_NAME, name);
        fTableOfSymbols[name] = symb;
    }
    else
        symb = fTableOfSymbols[name];

    return symb;
}

SgFunctionCallExp* createNewFCall(const char *name)
{
    SgSymbol *symb = createNewFunctionSymbol(name);
    return new SgFunctionCallExp(*symb);
}

void createNewFCall(SgExpression *expr, SgExpression *&retExp, const char *name, int nArgs)
{
    SgExpression *currArgs = ((SgFunctionCallExp *)expr)->args();
    SgExpression **Arg = new SgExpression*[nArgs];
    for (int i = 0; i < nArgs; ++i)
    {
        Arg[i] = currArgs->lhs();
        convertExpr(Arg[i], Arg[i]);
        currArgs = currArgs->rhs();
    }

    retExp = createNewFCall(name);
    if (nArgs != 0)
    {
        for (int i = 0; i < nArgs; ++i)
            ((SgFunctionCallExp*)retExp)->addArg(*Arg[i]);
    }
    else
        ((SgFunctionCallExp*)retExp)->addArg(*expr);
}

static SgExpression* convertDvmAssign(SgExpression *copy, const vector<pair<SgSymbol*, SgSymbol*> >& symbs)
{
    SgExpression* list = copy->lhs()->lhs();
    stack<SgExpression*> pointersToMul;
    while (list)
    {
        if (list->variant() == MULT_OP)
            pointersToMul.push(list);
        else if (list->rhs() && list->rhs()->variant() == MULT_OP)
            pointersToMul.push(list->rhs());
        list = list->lhs();
    }
    for (int z = 0; z < symbs.size(); ++z)
    {
        SgSymbol* curr = symbs[z].first;
        SgExpression* exp = pointersToMul.top();
        pointersToMul.pop();
        exp->setRhs(&(*exp->rhs() + *new SgVarRefExp(curr)));
    }
    return copy;
}

static SgForStmt* createFor(const vector<int>& dimSizes, const vector<pair<SgSymbol*, SgSymbol*> >& symbs, SgStatement *inner)
{
    SgForStmt* forSt = NULL;
    for (int z = 0; z < dimSizes.size(); ++z)
    {
        SgSymbol* s = symbs[z].first;
        SgSymbol* s_decl = symbs[z].second;

        SgExpression* start = &SgAssignOp(*new SgVarRefExp(*s_decl), *new SgValueExp(0));
        SgExpression* end = &(*new SgVarRefExp(*s) < *new SgValueExp(dimSizes[z]));
        SgExpression* step = new SgUnaryExp(PLUSPLUS_OP, *new SgVarRefExp(*s));

        forSt = new SgForStmt(start, end, step, forSt == NULL ? inner : forSt);
    }
    return forSt;
}

static pair<SgSymbol*, pair<vector<SgStatement*>, vector<SgStatement*> > > createForCopy(const vector<int> &dimSizes, SgExpression *dvmArray, bool in, bool out)
{    
    SgType* base = dvmArray->symbol()->type()->baseType();
    SgForStmt* forSt = NULL, *forStInv = NULL;
    SgStatement* inner = NULL;

    vector<SgStatement*> ret;
    vector<SgStatement*> retInv;

    vector<pair<SgSymbol*, SgSymbol*> > symbs(dimSizes.size());

    int total = 1;
    for (int z = 0; z < dimSizes.size(); ++z)
        total *= dimSizes[z];

    SgArrayType* arrT = new SgArrayType(*base);
    arrT->addDimension(new SgValueExp(total));

    char buf[256];
    sprintf(buf, "%d", arrayGenNum++);
    SgSymbol* array = new SgSymbol(VARIABLE_NAME, (string("_tfm_arr_") + buf).c_str(), arrT, NULL);

    for (int z = 0; z < dimSizes.size(); ++z)
    {
        sprintf(buf, "%d", z);
        SgSymbol* s = new SgSymbol(VARIABLE_NAME, (string("_tfm__") + buf).c_str());
        SgSymbol* s_decl = new SgSymbol(VARIABLE_NAME, (string("int _tfm__") + buf).c_str());
        symbs[z] = make_pair(s, s_decl);
    }

    SgArrayRefExp* arrayRef = new SgArrayRefExp(*array);
    SgExpression* subs = new SgVarRefExp(symbs[0].first);
    int dumS = 1;
    for (int z = 1; z < symbs.size(); ++z)
    {
        subs = &(*subs + (*new SgValueExp(dumS * dimSizes[symbs.size() - z]) * *new SgVarRefExp(symbs[1].first)));
        dumS *= dimSizes[symbs.size() - z];
    }
    
    SgExpression* copyDvmArrayElems = convertDvmAssign(&dvmArray->copy(), symbs);
    const string key(copyDvmArrayElems->unparse());

    if (autoTfmReplacing.find(key) != autoTfmReplacing.end())
        return make_pair(autoTfmReplacing[key], make_pair(ret, retInv));

    arrayRef->addSubscript(*subs);
    ret.push_back(makeSymbolDeclaration(array));

    if (in)
    {
        inner = new SgAssignStmt(*arrayRef, copyDvmArrayElems->copy());
        forSt = createFor(dimSizes, symbs, inner);
        ret.push_back(forSt);
    }

    if (out)
    {
        inner = new SgAssignStmt(copyDvmArrayElems->copy(), arrayRef->copy());
        forStInv = createFor(dimSizes, symbs, inner);
        retInv.push_back(forStInv);
    }

    autoTfmReplacing[key] = array;    
    return make_pair(array, make_pair(ret, retInv));
}

static vector<int> fillBitsOfArgs(SgProgHedrStmt *hedr)
{
    vector<int> bitsOfArgs;
    for (int z = 0; z < hedr->numberOfParameters(); ++z)
    {
        SgSymbol *par = hedr->parameter(z);
        int attr = par->attributes();
        if (attr & IN_BIT)
            bitsOfArgs.push_back(IN_BIT);
        else if (attr & OUT_BIT)
            bitsOfArgs.push_back(OUT_BIT);
        else 
            bitsOfArgs.push_back(INOUT_BIT);
    }

    return bitsOfArgs;
}

static bool isPrivate(const string& array)
{
    SgExpression* exp = private_list;
    while (exp)
    {
        if (exp->lhs()->symbol()->identifier() == array)
            return true;
        exp = exp->rhs();
    }
    return false;
}

//#define DEB
static bool matchPrototype(SgSymbol *funcSymb, SgExpression *&listArgs, bool isFunction)
{
    bool ret = true;

    const string name(funcSymb->identifier());

    vector<SgType*> *prototype = NULL;
    int num = 0;
    SgExpression* tmp = listArgs;
    while (tmp)
    {
        num++;
        tmp = tmp->rhs();
    }
    
    map <string, vector<vector<SgType*> > >::iterator it = interfaceProcedures.find(name);
    bool canFoundInterface = !(it == interfaceProcedures.end());

    //try to find function on current file
    //TODO: add support of many files
    //TODO: module functions with the same name
    vector<int> argsBits;
    if (canFoundInterface == false)
    {
#ifdef DEB
        map<string, vector<graph_node*>> tmp;
        for (graph_node* ndl = node_list; ndl; ndl = ndl->next)
            tmp[ndl->name].push_back(ndl);
#endif 
        for (graph_node *ndl = node_list; ndl; ndl = ndl->next)
        {
            if (ndl->name == name && current_file == ndl->file)
            {
                if (ndl->st_header == NULL)
                {
                    Error("Can not find procedure header %s", name.c_str(), 652, first_do_par);
                    ret = false;
                }
                else
                {
                    CreateIntefacePrototype(ndl->st_header);
                    argsBits = fillBitsOfArgs(isSgProgHedrStmt(ndl->st_header));
                }
            }
            else if(ndl->name == name && ndl->st_interface)
            {
                  CreateIntefacePrototype(ndl->st_interface);
                  argsBits = fillBitsOfArgs(isSgProgHedrStmt(ndl->st_interface));
            }
        }

        it = interfaceProcedures.find(name);
        canFoundInterface = !(it == interfaceProcedures.end());

        if (canFoundInterface == false)
        {
            Error("Can not find interface for procedure %s", name.c_str(), 653, first_do_par);
            ret = false;
        }            
    }
    else
    {
        for (graph_node* ndl = node_list; ndl; ndl = ndl->next)
            if (ndl->name == name && current_file == ndl->file)
                argsBits = fillBitsOfArgs(isSgProgHedrStmt(ndl->st_header));
    }
    
    if (canFoundInterface)
    {
        bool found = false;

        //TODO: add support of many interfaces with the same count of parameters
        for (int k = 0; k < it->second.size(); ++k)
        {
            if (it->second[k].size() == num)
            {
                found = true;
                prototype = &it->second[k];
                break;
            }
        }

        if (found == false)
        {
            Error("Can not find interface for procedure %s", name.c_str(), 653, first_do_par);
            ret = false;
        }
        else //Match here
        {
            SgExpression *argInCall = listArgs;
            for (int i = 0; i < num; ++i, argInCall = argInCall->rhs())
            {
                if (argInCall->lhs() == NULL)
                {
                    Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);
                    ret = false;
                    continue;
                }
                
                SgType *typeInCall;
                SgSymbol* parS = NULL;
                if (argInCall->lhs()->symbol()) // simple argument
                {
                    typeInCall = argInCall->lhs()->symbol()->type();
                    parS = argInCall->lhs()->symbol();
#ifdef DEB
                    printf("simple type of typeInCall %d, %s\n", typeInCall->variant(), argInCall->lhs()->symbol()->identifier());
#endif
                }
                else                      // expression
                {
                    typeInCall = argInCall->lhs()->type();
#ifdef DEB
                    printf("expression type of typeInCall %d\n", typeInCall->variant());
#endif
                }

                SgType *typeInProt = (*prototype)[i];
                SgType* typeInProtSave = (*prototype)[i];
                
                int countOfSubscrInCall = 0;
                int dimSizeInProt = 0;
                if (argInCall->lhs()->variant() == ARRAY_REF)
                {
                    SgExpression *subs = argInCall->lhs()->lhs();
                    while (subs)
                    {
                        countOfSubscrInCall++;
                        subs = subs->rhs();
                    }

                    SgArrayType* inCall = isSgArrayType(typeInCall);
                    SgArrayType* inProt = isSgArrayType(typeInProt);

                    if (countOfSubscrInCall == 0)
                    {
                        if (inCall == NULL || inProt == NULL) // inconsistency
                        {
                            if (isSgPointerType(typeInCall) && inProt)
                                typeInCall = typeInProt;
                            else
                            {
                                typeInCall = NULL;
#ifdef DEB
                                printf("typeInCall NULL 1\n");
#endif
                            }
                        }
                        else if (inCall->dimension() != inProt->dimension())
                        {
                            typeInCall = NULL;
#ifdef DEB
                            printf("typeInCall NULL 2\n");
#endif
                        }
                        else
                            typeInCall = typeInProt;
                    }
                    else
                    {
                        //TODO: not supported yet
                        if (inCall && inProt)
                        {
                            if (inCall->dimension() != inProt->dimension()) // TODO
                            {   //TODO: check for non distributed
                                typeInCall = typeInProt;
                                dimSizeInProt = inProt->dimension();
                            }
                            else
                            {
                                if (options.isOn(O_PL2) && dvm_parallel_dir->expr(0) == NULL)
                                    dimSizeInProt = inCall->dimension();

                                const int arrayDim = isPrivate(argInCall->lhs()->symbol()->identifier()) ? inCall->dimension() : 1;

                                if (isSgArrayType(typeInProt) && (!options.isOn(O_PL2) || dvm_parallel_dir->expr(0) != NULL)) // inconsistency
                                {
                                    if (inCall->dimension() == inProt->dimension())
                                    {
                                        typeInCall = typeInProt;
                                        dimSizeInProt = inProt->dimension();
                                    }
                                    else
                                    {
                                        typeInCall = NULL;
#ifdef DEB
                                        printf("typeInCall NULL 3\n");
#endif
                                    }
                                }
                                else if (arrayDim - countOfSubscrInCall == 0)
                                    typeInCall = typeInProt;
                                else // TODO
                                {
                                    typeInCall = NULL;
#ifdef DEB
                                    printf("typeInCall NULL 4\n");
#endif
                                }
                            }
                        }
                        else if (inProt) // inconsistency
                        {
                            typeInCall = NULL;
#ifdef DEB
                            printf("typeInCall NULL 5\n");
#endif
                        }
                        else if (inCall)
                        {
                            const int arrayDim = isPrivate(argInCall->lhs()->symbol()->identifier()) ? inCall->dimension() : 1;

                            if (arrayDim - countOfSubscrInCall == 0)
                                typeInCall = typeInProt;
                            else
                            {
                                typeInCall = NULL;
#ifdef DEB
                                printf("typeInCall NULL 6\n");
#endif
                            }
                        }
                    }
                }
                else
                {
                    if (typeInCall->variant() == T_DESCRIPT)
                        typeInCall = ((SgDescriptType*)typeInCall)->baseType();

                    if (typeInProt->variant() == typeInCall->variant())
                    {
                        if (typeInProt->hasBaseType() && !typeInCall->hasBaseType()) // inconsistency
                        {
                            typeInCall = NULL;
#ifdef DEB
                            printf("typeInCall NULL 7\n");
#endif
                        }

                        if (typeInProt->hasBaseType() && typeInCall)
                        {
                            if (typeInProt->baseType()->variant() != typeInCall->baseType()->variant()) // inconsistency
                            {
                                typeInCall = NULL;
#ifdef DEB
                                printf("typeInCall NULL 8\n");
#endif
                            }
                            else
                            {
                                typeInProt = typeInProt->baseType();
                                typeInCall = typeInCall->baseType();
                            }
                        }

                        if (typeInCall)
                        {
                            if (typeInProt->equivalentToType(typeInCall))
                                typeInCall = typeInProt;
                            else
                            {
                                if (typeInProt->length() && typeInCall->length())
                                {
                                    if (string(typeInProt->length()->unparse()) == string(typeInCall->length()->unparse()))
                                        typeInCall = typeInProt;
                                    else
                                    {
                                        typeInCall = NULL; // TODO
#ifdef DEB
                                        printf("typeInCall NULL 9\n");
#endif
                                    }
                                }
                                else if (typeInProt->selector() && typeInCall->selector())
                                {
                                    if (string(typeInProt->selector()->unparse()) == string(typeInCall->selector()->unparse()))
                                        typeInCall = typeInProt;
                                    else
                                    {
                                        typeInCall = NULL; // TODO
#ifdef DEB
                                        printf("typeInCall NULL 10\n");
#endif
                                    }
                                }
                                else
                                    ; //TODO
                            }
                        }

                        if (typeInProt != typeInCall)
                        {
                            if (CompareKind(typeInProt, typeInCall) != 1) // check selector
                            {
                                char buf[256];
                                sprintf(buf, "The type of %d argument of '%s' procedure can not be equal to actual parameter in call", i + 1, name.c_str());
                                Warning(buf, "", 655, first_do_par);
                            }
                            typeInCall = typeInProt;
                        }
                    }
                    else // check selector
                    {
                        if (CompareKind(typeInProt, typeInCall))
                            typeInCall = typeInProt;
                    }
                }

                if (typeInProt != typeInCall)
                {
                    char buf[256];
                    sprintf(buf, "Can not match the %d argument of '%s' procedure", i + 1, name.c_str());
                    Error(buf, "", 656, first_do_par);
                    ret = false;
                }
                else if (argInCall->lhs()->variant() == ARRAY_REF)
                {
                    if (countOfSubscrInCall == 0)
                    {
                        SgExpression *arr = argInCall->lhs();
                        SgType *type = arr->symbol()->type();

                        if (type->hasBaseType())
                            argInCall->setLhs(*new SgCastExp(*C_PointerType(C_Type(type->baseType())), *arr));
                        else
                            argInCall->setLhs(*new SgCastExp(*C_PointerType(C_Type(type)), *arr));
                    }
                    else
                    {
                        if (dimSizeInProt == 0)
                        {
                            if (isFunction)
                            {
                                SgExpression* arrayRef = argInCall->lhs();
                                convertExpr(arrayRef, arrayRef);
                            }
                        }
                        else
                        {
                            if (options.isOn(AUTO_TFM) && !isInPrivate(argInCall->lhs()->symbol()->identifier()))
                            {
                                //TODO: ranges, ex. (-1:2)

                                SgArrayType* arrT = isSgArrayType(typeInProtSave);
                                int dim = arrT->dimension();
                                vector<int> dimSizes(dim);
                                for (int z = 0; z < dim; ++z)
                                    dimSizes[z] = -1;

                                int dimTotal = 1;
                                for (int z = 0; z < dim; ++z)
                                {
                                    if (arrT->sizeInDim(z)->isInteger())
                                        dimTotal *= dimSizes[z] = arrT->sizeInDim(z)->valueInteger();
                                    else
                                        dimTotal = -1;
                                }

                                if (dimTotal != -1)
                                {
                                    std::reverse(dimSizes.begin(), dimSizes.end());
                                    bool ifIn = true;
                                    bool ifOut = true;

                                    pair<SgSymbol*, pair<vector<SgStatement*>, vector<SgStatement*> > > conv = createForCopy(dimSizes, argInCall->lhs(), ifIn, ifOut);

                                    if ( (argsBits[i] & IN_BIT) || (argsBits[i] & INOUT_BIT))
                                        for (int z = 0; z < conv.second.first.size(); ++z)
                                            insertBefore[curTranslateStmt].push_back(conv.second.first[z]);

                                    if ((argsBits[i] & OUT_BIT) || (argsBits[i] & INOUT_BIT))
                                        for (int z = 0; z < conv.second.second.size(); ++z)
                                            insertAfter[curTranslateStmt].push_back(conv.second.second[z]);

                                    argInCall->setLhs(*new SgArrayRefExp(*conv.first));
                                }
                                else
                                {
                                    char buf[256];
                                    sprintf(buf, "Unsupported variant of '%s' procedure call", name.c_str());
                                    Error(buf, "", 657, first_do_par);
                                }
                            }
                            else
                            {
                                SgExpression* arr = argInCall->lhs();

                                if (options.isOn(O_PL2))
                                {
                                    SgType* cast = NULL;
                                    if (typeInProtSave->hasBaseType())
                                        cast = C_PointerType(C_Type(typeInProtSave->baseType()));
                                    else
                                        cast = C_PointerType(C_Type(typeInProtSave));
                                    argInCall->setLhs(*new SgCastExp(*cast, SgAddrOp(*arr)));
                                }
                                else
                                    argInCall->setLhs(SgAddrOp(*arr));
                            }
                        }
                    }
                }
                else
                {
                    SgExpression* arg = argInCall->lhs();
                    SgType* orig = arg->type();
                    SgType* typeCopy = orig->copyPtr();

                    SgExpression* selector = typeCopy->selector();
                    if (selector)
                    {
                        typeCopy->deleteSelector();
                        arg->setType(typeCopy);
                    }

                    if (isFunction)
                        convertExpr(arg, arg);

                    if (selector)
                    {
                        int size = -1;
                        SgExpression* e2 = TypeKindExpr(orig);
                        if (e2 && e2->isInteger())
                            size = e2->valueInteger();

                        if (size > 0)
                        {
                            const int var = typeCopy->variant();
                            if (var == T_FLOAT || var == T_DOUBLE)
                            {
                                if (size == 4)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "float"), *new SgExprListExp(*arg));
                                else if (size == 8)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "double"), *new SgExprListExp(*arg));
                            }
                            else if (var == T_INT || var == T_BOOL)
                            {
                                if (size == 1)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "char"), *new SgExprListExp(*arg));
                                else if (size == 2)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "short"), *new SgExprListExp(*arg));
                                else if (size == 4)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "int"), *new SgExprListExp(*arg));
                                else if (size == 8)
                                    arg = new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "long long"), *new SgExprListExp(*arg));
                            }
                        }
                    }
                    
                    argInCall->setLhs(arg);
                }
            }
        }
    }

    return ret;
}

void convertExpr(SgExpression *expr, SgExpression* &retExp)
{
    if (expr)
    {
        int var = expr->variant();
        SgExpression *lhs = NULL, *rhs = NULL;

        if (var != FUNC_CALL)
        {
            if (expr->lhs())
            {
                lhs = expr->lhs();
                convertExpr(lhs, lhs);
            }

            if (expr->rhs())
            {
                rhs = expr->rhs();
                convertExpr(rhs, rhs);
            }
        }

        if (var == EXP_OP)
        {
            bool default_ = false;

            if (rhs->variant() == INT_VAL)
            {
                int i = rhs->valueInteger();
                if (i == 0)
                    retExp = new SgValueExp(1);
                else if (i == 1)
                    retExp = lhs;
                else if (i == 2)
                {
                    if (lhs->variant() != FUNC_CALL && lhs->variant() != PROC_CALL)
                        retExp = &(*lhs * *lhs);
                    else
                        default_ = true;
                }
                else
                    default_ = true;
            }
            else
                default_ = true;

            if (default_)
            {
                SgFunctionCallExp *tmpF = new SgFunctionCallExp(*createNewFunctionSymbol("pow"));
                tmpF->addArg(*lhs);
                tmpF->addArg(*rhs);
                retExp = tmpF;
            }
        }
        else if(var == RECORD_REF)
            retExp = expr;
        else if (var == FUNC_CALL)
        {
            SgFunctionCallExp *tmpF = (SgFunctionCallExp *)expr;
            const char *name = tmpF->funName()->identifier();
            map<string, FunctionParam>::iterator it = handlersOfFunction.find(name);
            if (!strcmp(name, "present"))
            {
               /* string argName = expr->lhs()->lhs()->symbol()->identifier();
                SgStatement* funcHdr = curTranslateStmt;
                SgExpression* newPresent = makePresentExpr(argName,funcHdr);
                retExp = newPresent;*/
                SgExpression* pres = new SgExpression(RECORD_REF);
                pres->setLhs(new SgVarRefExp(expr->lhs()->lhs()->symbol()));
                pres->setRhs(new SgVarRefExp(*new SgSymbol(FIELD_NAME, "isExist")));
                retExp = pres;
            }
            else if(!strcmp(name, "ub"))
                retExp = expr;
            else
            {
                if (it != handlersOfFunction.end())
                    it->second.CallHandler(expr, retExp);
                else
                {
                    SgSymbol *symb = tmpF->funName();
                    SgStatement *inter = getInterfaceForCall(symb);
                    if(inter)
                    {
                        //switch arguments by keyword
                        expr = switchArgumentsByKeyword(name, tmpF, inter);
                        //check ommited arguments
                        //transform fact to formal
                    }

                    SgExpression *tmp = expr->lhs();
                    matchPrototype(tmpF->funName(), tmp, true);

                    retExp->setLhs(expr->lhs());
                    retExp->setRhs(expr->rhs());

                    if (isUserFunction(tmpF->funName()) == 0)
                    {
                        printf("  [EXPR ERROR: %s, line %d, user line %d] unsupported variant of func call with name \"%s\"\n", __FILE__, __LINE__, first_do_par->lineNumber(), name);
                        if (unSupportedVars.size() != 0)
                            Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);
                    }
                }
            }
        }
        else if (var == DOUBLE_VAL)
        {
            char *digit_o = ((SgValueExp*)expr)->doubleValue();
            SgExpression *val = ((SgValueExp*)expr)->type()->selector();

            char *digit = new char[strlen(digit_o) + 1];
            strcpy(digit, digit_o);
            for (size_t i = 0; i < strlen(digit); ++i)
            {
                if (digit[i] == 'd')
                {
                    digit[i] = 'e';
                    break;
                }
            }
            SgValueExp *valDouble = new SgValueExp(double(0.0), digit);
            delete[]digit;

            if (val != NULL)
            {
                if (val->valueInteger() == 8) // double
                    createNewFCall(valDouble, retExp, "double", 0);
                else if (val->valueInteger() == 4) // float
                    createNewFCall(valDouble, retExp, "float", 0);
                else
                    retExp = valDouble;
            }
            else
                retExp = valDouble;
        }
        else if (var == FLOAT_VAL)
        {
            char *digit_o = ((SgValueExp*)expr)->floatValue();
            SgExpression *val = ((SgValueExp*)expr)->type()->selector();

            char *digit = new char[strlen(digit_o) + 2];
            strcpy(digit, digit_o);
            digit[strlen(digit_o)] = 'f';
            digit[strlen(digit_o) + 1] = '\0';

            SgValueExp *valFloat = new SgValueExp(float(0.0), digit);
            delete[]digit;

            if (val != NULL)
            {
                if (val->valueInteger() == 8) // double
                    createNewFCall(valFloat, retExp, "double", 0);
                else if (val->valueInteger() == 4) // float
                    createNewFCall(valFloat, retExp, "float", 0);
                else
                    retExp = valFloat;
            }
            else
                retExp = valFloat;
        }
        else if (var == INT_VAL)
        {
            SgExpression *val = ((SgValueExp*)expr)->type()->selector();
            int digit = ((SgValueExp*)expr)->valueInteger();
            if (val != NULL)
            {
                if (val->valueInteger() == 8) // long
                    createNewFCall(new SgValueExp(digit), retExp, "long", 0);
                else if (val->valueInteger() == 4) // int
                    createNewFCall(new SgValueExp(digit), retExp, "int", 0);
                else if (val->valueInteger() == 2) // short
                    createNewFCall(new SgValueExp(digit), retExp, "short", 0);
                else if (val->valueInteger() == 1) // char
                    createNewFCall(new SgValueExp(digit), retExp, "char", 0);
                else
                    retExp = expr;
            }
            else
                retExp = expr;
        }
        else if (var == COMPLEX_VAL)
        {
            SgValueExp *tmp = ((SgValueExp*)expr);
            SgExpression *re = ((SgValueExp*)expr)->realValue();
            SgExpression *im = ((SgValueExp*)expr)->imaginaryValue();

            int kind = 8;
            if (re->variant() != DOUBLE_VAL && im->variant() != DOUBLE_VAL)
                kind = 4;

            if (kind == 8)
                retExp = new SgFunctionCallExp(*createNewFunctionSymbol("dcmplx2"));
            else
                retExp = new SgFunctionCallExp(*createNewFunctionSymbol("cmplx2"));

            convertExpr(re, re);
            convertExpr(im, im);

            ((SgFunctionCallExp*)retExp)->addArg(*re);
            ((SgFunctionCallExp*)retExp)->addArg(*im);
        }
        else if (var == ARRAY_REF)
        {
            bool ifInPrivateList = false;
            size_t idx = 0;

            char *strName = expr->symbol()->identifier();
            for (; idx < arrayInfo.size(); ++idx)
            {
                if (arrayInfo[idx].name == strName)
                {
                    ifInPrivateList = true;
                    break;
                }
            }

            if (ifInPrivateList)
            {
                int dim = isSgArrayType(expr->symbol()->type())->dimension();

                if (dim > 0 && expr->lhs()) // DIM > 0 && ARRAY_REF is not under CALL
                {
                    stack<SgExpression*> allArraySub;
                    //swap subscripts and correct exps

                    SgExpression *tmp = expr->lhs();
                    for (int i = 0; i < dim; ++i)
                    {
                        SgExpression *conv = tmp->lhs();
                        convertExpr(conv, conv);
                        tmp = tmp->rhs();
                        allArraySub.push(conv);
                    }

                    tmp = expr->lhs();
                    int k = 0;
                    for (int i = 0; i < dim; ++i)
                    {
                        if (arrayInfo[idx].correctExp[dim - 1 - k])
                            tmp->setLhs(*allArraySub.top() - *arrayInfo[idx].correctExp[dim - 1 - k]);
                        else
                            tmp->setLhs(*allArraySub.top());
                        allArraySub.pop();
                        k++;
                        tmp = tmp->rhs();
                    }


                    if (arrayInfo[idx].typeRed == 1)
                    {
                        // revert order of subscr
                        stack<SgExpression*> allArraySub;
                        SgExpression *tmp = expr->lhs();
                        for (int i = 0; i < dim; ++i)
                        {
                            allArraySub.push(&tmp->lhs()->copy());
                            tmp = tmp->rhs();
                        }

                        tmp = expr->lhs();
                        for (int i = 0; i < dim; ++i)
                        {
                            tmp->setLhs(*allArraySub.top());
                            allArraySub.pop();
                            tmp = tmp->rhs();
                        }

                        // linearized red arrays
                        expr->setLhs(LinearFormForRedArray(expr->symbol(), expr->lhs(), arrayInfo[idx].rsl));
                    }
                }
            }
            // else global or dvm array
            retExp = expr;
        }
        else if (var == VAR_REF)
            retExp = &expr->copy();
        else if (var == NEQV_OP)
        {
#ifdef INTEL_LOGICAL_TYPE
            retExp  = new SgExpression(XOR_OP, lhs, rhs);
#else
            retExp = &(*lhs != *rhs);
#endif
        }
        else if (var == EQV_OP)
        {
#ifdef INTEL_LOGICAL_TYPE
            retExp = new SgExpression(BIT_COMPLEMENT_OP, new SgExpression(XOR_OP, lhs, rhs), NULL);
#else
        retExp = &(*lhs == *rhs);
#endif
        }
        else if (var == AND_OP)
            retExp = new SgExpression(BITAND_OP, lhs, rhs);
        else if (var == OR_OP)
            retExp = new SgExpression(BITOR_OP, lhs, rhs);
        else if (var == NOT_OP)
        {
#ifdef INTEL_LOGICAL_TYPE
            retExp = new SgExpression(BIT_COMPLEMENT_OP, lhs, NULL);
#else
            retExp = new SgExpression(NE_OP, lhs, new SgKeywordValExp("true"));
#endif
        }
        else if (var == BOOL_VAL)
        {         
            bool val = ((SgValueExp*)expr)->boolValue();
#ifdef INTEL_LOGICAL_TYPE
            retExp = val ? new SgExpression(BIT_COMPLEMENT_OP, new SgValueExp(0), NULL) : new SgValueExp(0);
#else
            retExp = new SgKeywordValExp(val ? "true" : "false");
#endif
        }
        else
        {
            // known vars: ADD_OP, SUBT_OP, MULT_OP, DIV_OP, MINUS_OP, UNARY_ADD_OP, CONST_REF, EXPR_LIST, 
            retExp->setLhs(lhs);
            retExp->setRhs(rhs);
            if (supportedVars.find(var) == supportedVars.end())
                unSupportedVars.insert(var);
        }
    }
}

static SgExpression* convertReductionAddressForAtomic(SgExpression* exp)
{
    SgExpression* ref = exp->copyPtr();
    ref->setLhs(NULL);

    SgExpression* idx = exp->lhs()->copyPtr();

    return new SgExpression(ADD_OP, ref, idx);
}

//TODO: need to check bitwise operations
static SgExpression* splitReductionForAtomic(SgExpression* lhs, SgExpression* rhs, const int num_red)
{
    SgExpression* args = NULL;
    if (!lhs || !rhs)
    {
        Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);
        return NULL;
    }

    string left(lhs->unparse());
    set<int> op;
    if (num_red == 1) // sum
    {
        op.insert(ADD_OP);
        op.insert(SUBT_OP);
    }
    else if (num_red == 2)  // product
        op.insert(MULT_OP);
    else if (num_red == 3)  // max
        op.insert(FUNC_CALL);
    else if (num_red == 4)  // min
        op.insert(FUNC_CALL);
    else if (num_red == 5)  // and
        op.insert(BITAND_OP);
    else if (num_red == 6)  // or
        op.insert(BITOR_OP);
    else if (num_red == 7)  // neqv
        op.insert(XOR_OP);
    else if (num_red == 8)  // eqv
    {
        if (rhs->variant() == BIT_COMPLEMENT_OP)
            rhs = rhs->lhs();
        op.insert(XOR_OP);
    }

    if (op.size())
    {
        if (op.find(rhs->variant()) != op.end())
        {
            SgExpression* l_part = rhs->lhs();
            SgExpression* r_part = rhs->rhs();
            if (rhs->variant() == FUNC_CALL)
            {
                if (rhs->lhs())
                {
                    if (rhs->lhs()->lhs())
                        l_part = rhs->lhs()->lhs();
                    if (rhs->lhs()->rhs() && rhs->lhs()->rhs()->lhs())
                        r_part = rhs->lhs()->rhs()->lhs();
                }
            }

            if (l_part && r_part)
            {
                string Lpart(l_part->unparse());
                string Rpart(r_part->unparse());

                bool ok = false;
                if (Lpart == left)
                    ok = true;
                else if (Rpart == left)
                {
                    std::swap(l_part, r_part);
                    ok = true;
                }

                if (ok)
                {
                    if (rhs->variant() == SUBT_OP)
                        r_part = new SgExpression(MINUS_OP, r_part, NULL);

                    SgExpression* arg1 = convertReductionAddressForAtomic(l_part);
                    SgExpression* arg2 = r_part;

                    args = new SgExpression(EXPR_LIST, arg1, new SgExpression(EXPR_LIST, arg2, NULL));
                }
            }
        }
    }

    if (args == NULL)
    {
        string right(rhs->unparse());
        Error("Can not match reduction template for this pattern: %s", (left + " = " + right).c_str(), 658, first_do_par);
    }

    return args;
}

static bool convertStmt(SgStatement* &st, pair<SgStatement*, SgStatement*> &retSts, vector < stack < SgStatement*> > &copyBlock, 
                        int countOfCopy, int lvl, const map<string, int>& redArraysWithUnknownSize)
{
    bool needReplace = false;
    SgStatement *labSt = NULL;
    SgStatement *retSt = NULL;
    curTranslateStmt = st;
    if (st->hasLabel())
    {
        if (lvl == 0)
            convertLabel(st, labSt, false);
        else
            convertLabel(st, labSt, true);

        for (int i = 0; i < countOfCopy; ++i)
            copyBlock[i].push(&st->lexPrev()->copy());
    }

    if (st->variant() == ASSIGN_STAT)
    {
        SgExpression *lhs = st->expr(0);
        SgExpression *rhs = st->expr(1);

#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert assign node\n");
        lvl_convert_st += 2;
#endif
        convertExpr(lhs, lhs);
        convertExpr(rhs, rhs);
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert assign node\n");
#endif
        if (lhs->variant() == ARRAY_REF && redArraysWithUnknownSize.find(lhs->symbol()->identifier()) != redArraysWithUnknownSize.end())
        {
            const string arrayName = lhs->symbol()->identifier();
            const int num_red = redArraysWithUnknownSize.find(arrayName)->second;
            string atomicName = "NULL";

            if (num_red == 1) // sum
                atomicName = "__dvmh_atomic_add";
            else if (num_red == 2)  // product
                atomicName = "__dvmh_atomic_prod";
            else if (num_red == 3)  // max
                atomicName = "__dvmh_atomic_max";
            else if (num_red == 4)  // min
                atomicName = "__dvmh_atomic_min";
            else if (num_red == 5)  // and
                atomicName = "__dvmh_atomic_and";
            else if (num_red == 6)  // or
                atomicName = "__dvmh_atomic_or";
            else if (num_red == 7)  // neqv
                atomicName = "__dvmh_atomic_neqv";
            else if (num_red == 8)  // eqv
                atomicName = "__dvmh_atomic_eqv";

            if (atomicName == "NULL")
            {
                Error("Unsupported reduction type by unknown(large) array size", "", 659, first_do_par);
                retSt = new SgCExpStmt(SgAssignOp(*lhs, *rhs));
            }
            else
            {
                SgFunctionSymb* fCall = new SgFunctionSymb(FUNCTION_NAME, atomicName.c_str(), *SgTypeInt(), *kernel_st);

                SgExpression* args = splitReductionForAtomic(lhs, rhs, num_red);
                if (args)
                    retSt = new SgCExpStmt(*new SgFunctionCallExp(*fCall, *args));
            }
        }
        else
            retSt = new SgCExpStmt(SgAssignOp(*lhs, *rhs));
        needReplace = true;        
    }
    else if (st->variant() == CONT_STAT)
    {
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert continue node\n");
        lvl_convert_st += 2;
#endif
        retSt = NULL;
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert continue node\n");

#endif
        needReplace = true;
    }
    else if (st->variant() == ARITHIF_NODE)
    {
        SgExpression *cond = st->expr(0);
        SgExpression *lb = st->expr(1);
        SgLabel *arith_lab[3];
        int i = 0;
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert arithif node\n");
        lvl_convert_st += 2;
#endif
        convertExpr(cond, cond);
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert arithif node\n");
#endif
        while (lb)
        {
            SgLabel *lab = ((SgLabelRefExp *)(lb->lhs()))->label();
            SgStatement *labRet = NULL;

            long lab_num = lab->thelabel->stateno;
            labels_num.insert(lab_num);

            createNewLabel(labRet, lab);
            arith_lab[i] = ((SgLabelRefExp *)(lb->lhs()))->label();
            i++;
            lb = lb->rhs();
        }


        retSt = new SgIfStmt(*cond < *new SgValueExp(0), *new SgGotoStmt(*arith_lab[0]),
            *new SgIfStmt(SgEqOp(*cond, *new SgValueExp(0)), *new SgGotoStmt(*arith_lab[1]), *new SgGotoStmt(*arith_lab[2])));
        needReplace = true;
    }
    else if (st->variant() == LOGIF_NODE)
    {
        SgExpression *cond = st->expr(0);
        convertExpr(cond, cond);
        SgStatement *body = ((SgLogIfStmt*)st)->body();
        pair<SgStatement*, SgStatement*> t;
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert logicif node\n");
        lvl_convert_st += 2;
#endif
        convertStmt(body, t, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert logicif node\n");
#endif
        retSt = new SgIfStmt(*cond, *t.first);
        if (t.second)
            labSt = t.second;
        needReplace = true;
    }
    else if (st->variant() == IF_NODE)
    {
        SgStatement *tb = ((SgIfStmt*)st)->trueBody();
        SgStatement *fb = ((SgIfStmt*)st)->falseBody();
        SgIfStmt *newIfSt = NULL;

        if (!fb)
        {
            SgStatement *tmp = st->lexNext();
            stack<SgStatement*> bodySts;
            while (st->lastNodeOfStmt() != tmp)
            {
                pair<SgStatement*, SgStatement*> convSt;
#if TRACE
                printfSpaces(lvl_convert_st);
                printf("convert if node\n");
                lvl_convert_st += 2;
#endif
                convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
                lvl_convert_st-=2;
                printfSpaces(lvl_convert_st);
                printf("end of convert if node\n");
#endif
                if (convSt.second)
                    bodySts.push(convSt.second);
                if (convSt.first)
                    bodySts.push(convSt.first);

                setControlLexNext(tmp);
            }

            if (tmp->variant() == CONTROL_END)
            {
                pair<SgStatement*, SgStatement*> convSt;
                convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
                if (convSt.second)
                    bodySts.push(convSt.second);
            }

            SgExpression *cond = ((SgIfStmt*)st)->conditional();
            convertExpr(cond, cond);
            if (bodySts.size())
            {
                retSt = new SgIfStmt(*cond, *bodySts.top());
                bodySts.pop();
            }
            else
                retSt = new SgIfStmt(*cond, *new SgStatement(1), 2);

            int size = bodySts.size();
            for (int i = 0; i < size; ++i)
            {
                retSt->insertStmtAfter(*bodySts.top());
                bodySts.pop();
            }
            needReplace = true;
        }
        else
        {
            stack<stack<SgStatement*> > bodySts;
            stack<SgStatement*> bodyFalse;
            stack<SgExpression*> conds;
            SgStatement *fb_ControlEnd = NULL;

            stack<SgStatement*> t;
            SgExpression *cond = ((SgIfStmt*)st)->conditional();
            convertExpr(cond, cond);
            conds.push(cond);
            for (;;)
            {
                if (fb->variant() == ELSEIF_NODE)
                {
                    if (((SgIfStmt*)fb)->falseBody())
                    {
                        if (((SgIfStmt*)fb)->falseBody()->variant() == ELSEIF_NODE)
                            fb = ((SgIfStmt*)fb)->falseBody();
                        else
                        {
                            fb = ((SgIfStmt*)fb)->falseBody();
                            fb_ControlEnd = fb->controlParent()->lastNodeOfStmt();
                            break;
                        }
                    }
                    else
                    {
                        fb = fb->lastNodeOfStmt();
                        fb_ControlEnd = fb;
                        break;
                    }
                }
                else
                {
                    fb_ControlEnd = fb;
                    while (fb_ControlEnd->variant() != CONTROL_END)
                        setControlLexNext(fb_ControlEnd);
                    break;
                }
            }

            if (tb == NULL)
                tb = ((SgIfStmt*)st)->falseBody();

            while (tb != fb)
            {
                if (tb->variant() == ELSEIF_NODE)
                {
                    bodySts.push(t);
                    SgExpression *cond = ((SgIfStmt*)tb)->conditional();
                    convertExpr(cond, cond);
                    conds.push(cond);
                    t = stack<SgStatement*>();
                    tb = tb->lexNext();
                }
                else if (tb->variant() != CONTROL_END)
                {
                    pair<SgStatement*, SgStatement*> tmp;
#if TRACE
                    printfSpaces(lvl_convert_st);
                    printf("convert if node\n");
                    lvl_convert_st += 2;
#endif
                    convertStmt(tb, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
                    lvl_convert_st-=2;
                    printfSpaces(lvl_convert_st);
                    printf("end of convert if node\n");
#endif
                    if (tmp.second)
                        t.push(tmp.second);
                    if (tmp.first)
                        t.push(tmp.first);

                    setControlLexNext(tb);
                }
                else
                    tb = tb->lexNext();
            }
            bodySts.push(t);

            while (fb != fb_ControlEnd)
            {
                pair<SgStatement*, SgStatement*> tmp;
#if TRACE
                printfSpaces(lvl_convert_st);
                printf("convert if node\n");
                lvl_convert_st += 2;
#endif
                convertStmt(fb, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
                lvl_convert_st-=2;
                printfSpaces(lvl_convert_st);
                printf("end of convert if node\n");
#endif
                if (tmp.second)
                    bodyFalse.push(tmp.second);
                if (tmp.first)
                    bodyFalse.push(tmp.first);

                setControlLexNext(fb);
            }

            if (fb->variant() == CONTROL_END)
            {
                pair<SgStatement*, SgStatement*> tmp;
                convertStmt(fb, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
                if (tmp.second)
                    bodyFalse.push(tmp.second);
            }

            if (bodyFalse.size())
            {
                if (bodySts.top().size() != 0)
                    newIfSt = new SgIfStmt(*conds.top(), *bodySts.top().top(), *bodyFalse.top());
                else
                    newIfSt = new SgIfStmt(*conds.top(), *bodyFalse.top(), 0);

                bodyFalse.pop();
                int cond1 = bodyFalse.size();
                for (int i = 0; i < cond1; ++i)
                {
                    newIfSt->falseBody()->insertStmtBefore(*bodyFalse.top(), *newIfSt);
                    bodyFalse.pop();
                }
            }
            else
            {
                if (bodySts.top().size())
                    newIfSt = new SgIfStmt(*conds.top(), *bodySts.top().top()); // !!!!
                else
                    newIfSt = new SgIfStmt(*conds.top(), *new SgStatement(1), 2); // !!!!
            }

            conds.pop();
            int cond1 = bodySts.size();
            for (int i = 0; i < cond1; ++i)
            {
                stack<SgStatement*> tmpS = bodySts.top();
                int cond2;
                bodySts.pop();
                if (i == 0)
                {
                    if (tmpS.size() != 0)
                    {
                        tmpS.pop();
                        cond2 = tmpS.size();
                        for (int k = 0; k < cond2; ++k)
                        {
                            newIfSt->insertStmtAfter(*tmpS.top(), *newIfSt);
                            tmpS.pop();
                        }
                    }
                }
                else
                {
                    if (tmpS.size() != 0)
                    {
                        newIfSt = new SgIfStmt(*conds.top(), *tmpS.top(), *newIfSt);
                        conds.pop();
                        tmpS.pop();
                        cond2 = tmpS.size();
                        for (int k = 0; k < cond2; ++k)
                        {
                            newIfSt->insertStmtAfter(*tmpS.top(), *newIfSt);
                            tmpS.pop();
                        }
                    }
                    else
                    {
                        newIfSt = new SgIfStmt(*conds.top(), *newIfSt, 0);
                        conds.pop();
                    }
                }
            }

            retSt = newIfSt;
            needReplace = true;
        }
    }
    else if (st->variant() == FOR_NODE)
    {
        SgSymbol *cycleName = NULL;
        if (isSgVarRefExp(st->expr(2)))
            cycleName = isSgVarRefExp(st->expr(2))->symbol();

        SgSymbol *it = ((SgForStmt *)st)->symbol();
        SgExpression *ex1 = ((SgForStmt *)st)->start();
        SgExpression *ex2 = ((SgForStmt *)st)->end();
        SgExpression *ex3 = NULL;
        int ex3_lav = 0;
        SgStatement *inDo = ((SgForStmt *)st)->body();
        SgSymbol *cond = new SgSymbol(VARIABLE_NAME, getNestCond());
        SgSymbol *newVar = new SgSymbol(VARIABLE_NAME, getNewCycleVar(it->identifier()));
        SgFunctionCallExp *abs_f = new SgFunctionCallExp(*createNewFunctionSymbol("abs"));
        SgFunctionCallExp *abs_f1 = new SgFunctionCallExp(*createNewFunctionSymbol("abs"));
        stack<SgStatement*> bodySt;


        if (((SgForStmt *)st)->step())
            ex3 = ((SgForStmt *)st)->step();
        else
        {
            ex3 = new SgValueExp(1);
            ex3_lav = 1;
        }

        SgStatement *lastNode = ((SgForStmt *)st)->lastNodeOfStmt();

        while (inDo != lastNode)
        {
            pair<SgStatement*, SgStatement*> tmp;
#if TRACE
            printfSpaces(lvl_convert_st);
            printf("convert for node\n");
            lvl_convert_st += 2;
#endif
            map<SgStatement*, vector<SgStatement*> > save_insertBefore, save_insertAfter;            
            saveInsertBeforeAfter(save_insertAfter, save_insertBefore);

            convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
            lvl_convert_st-=2;
            printfSpaces(lvl_convert_st);
            printf("end of convert for node\n");
#endif
            copyToStack(bodySt, insertBefore);
            if (tmp.second)
                bodySt.push(tmp.second);
            if (tmp.first)
                bodySt.push(tmp.first);
            copyToStack(bodySt, insertAfter);
            
            restoreInsertBeforeAfter(save_insertAfter, save_insertBefore);            
            setControlLexNext(inDo);
        }

        if (lastNode->variant() != CONTROL_END)
        {
            pair<SgStatement*, SgStatement*> tmp;
#if TRACE
            printfSpaces(lvl_convert_st);
            printf("convert for node\n");
            lvl_convert_st += 2;
#endif
            map<SgStatement*, vector<SgStatement*> > save_insertBefore, save_insertAfter;
            saveInsertBeforeAfter(save_insertAfter, save_insertBefore);
            convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
            lvl_convert_st-=2;
            printfSpaces(lvl_convert_st);
            printf("end of convert for node\n");
#endif
            copyToStack(bodySt, insertBefore);
            if (tmp.second)
                bodySt.push(tmp.second);
            if (tmp.first)
                bodySt.push(tmp.first);
            copyToStack(bodySt, insertAfter);
            restoreInsertBeforeAfter(save_insertAfter, save_insertBefore);
        }
        else
        {
            pair<SgStatement*, SgStatement*> tmp;

            map<SgStatement*, vector<SgStatement*> > save_insertBefore, save_insertAfter;
            saveInsertBeforeAfter(save_insertAfter, save_insertBefore);
            convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
            copyToStack(bodySt, insertBefore);
            if (tmp.second)
                bodySt.push(tmp.second);
            copyToStack(bodySt, insertAfter);
            restoreInsertBeforeAfter(save_insertAfter, save_insertBefore);
        }

        SgExprListExp *tt = new SgExprListExp();
        SgExprListExp *tt1 = new SgExprListExp();
        SgExprListExp *tt2 = new SgExprListExp();
        SgExprListExp *tt3 = new SgExprListExp();

        tt->setLhs(SgAssignOp(*new SgVarRefExp(it), *ex1));

        abs_f->addArg(*ex3);
        abs_f1->addArg(*ex1 - *ex2);

        // IF EXPR: t_ex1 ? t_ex2 : t_ex3
        SgExpression *t_ex1 = &(*ex1 > *ex2 && *ex3 > *new SgValueExp(0) || *ex1 < *ex2 && *ex3 < *new SgValueExp(0));
        SgExpression *t_ex2 = &SgAssignOp(*new SgVarRefExp(cond), *new SgValueExp(-1));
        SgExpression *t_ex3;
        if (ex3_lav != 1)
            t_ex3 = &SgAssignOp(*new SgVarRefExp(cond), (*abs_f1 + *abs_f) / *abs_f);
        else
            t_ex3 = &SgAssignOp(*new SgVarRefExp(cond), (*abs_f1 + *abs_f));

        tt1->setLhs(*new SgExprIfExp(*t_ex1, *t_ex2, *t_ex3));
        tt->setRhs(tt1);
        tt2->setLhs(SgAssignOp(*new SgVarRefExp(*newVar), *new SgValueExp(0)));
        tt1->setRhs(tt2);
        tt3->setLhs(&SgAssignOp(*new SgVarRefExp(it), *new SgVarRefExp(it) + *ex3));
        tt3->setRhs(new SgExprListExp());
        tt3->rhs()->setLhs(&SgAssignOp(*new SgVarRefExp(newVar), *new SgVarRefExp(newVar) + *new SgValueExp(1)));

        if (SAPFOR_CONV) // TODO: negative step
        {
            SgExprListExp* start = new SgExprListExp();
            start->setLhs(SgAssignOp(*new SgVarRefExp(it), *ex1));

            SgExprListExp* step = new SgExprListExp();
            step->setLhs(&SgAssignOp(*new SgVarRefExp(it), *new SgVarRefExp(it) + *ex3));
            retSt = new SgForStmt(start, &(*new SgVarRefExp(it) <= ex2->copy()), step, NULL);
        }
        else
            retSt = new SgForStmt(tt, &(*new SgVarRefExp(*newVar) < *new SgVarRefExp(cond)), tt3, NULL);

        if (cycleName)
        {
            vector<SgLabel*> labs;
            vector<SgStatement*> labsSt;
            createNewLabel(labsSt, labs, cycleName->identifier());

            bodySt.push(labsSt[0]);
            labels_num.insert(labs[0]->thelabel->stateno);
            bodySt.push(new SgContinueStmt());

            bodySt.push(labsSt[1]);
            labels_num.insert(labs[1]->thelabel->stateno);
            bodySt.push(new SgBreakStmt());
        }

        int sizeStack = bodySt.size();
        for (int i = 0; i < sizeStack; ++i)
        {
            retSt->insertStmtAfter(*bodySt.top(), *retSt);
            bodySt.pop();
        }
        newVars.push_back(cond);

        SgExprListExp *e = new SgExprListExp(*new SgVarRefExp(cond));
        e->setRhs(private_list);
        private_list = e;

        bool needToadd = true;
        for (size_t i = 0; i < newVars.size(); ++i)
        {
            if (strcmp(newVars[i]->identifier(), newVar->identifier()) == 0)
            {
                needToadd = false;
                break;
            }
        }
        if (needToadd)
        {
            newVars.push_back(newVar);
            e = new SgExprListExp(*new SgVarRefExp(newVar));
            e->setRhs(private_list);
            private_list = e;
        }

        needReplace = true;
    }
    else if (st->variant() == WHILE_NODE)
    {
        SgSymbol *cycleName = NULL;
        if (isSgVarRefExp(st->expr(2)))
            cycleName = isSgVarRefExp(st->expr(2))->symbol();

        SgExpression *conditional = ((SgWhileStmt *)st)->conditional();
        stack<SgStatement*> bodySt;
        SgStatement *inDo = ((SgWhileStmt *)st)->body();
        SgStatement *lastNode = ((SgWhileStmt *)st)->lastNodeOfStmt();


        while (inDo != lastNode)
        {
            pair<SgStatement*, SgStatement*> tmp;
#if TRACE
            printfSpaces(lvl_convert_st);
            printf("convert while node\n");
            lvl_convert_st += 2;
#endif
            (void)convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
            lvl_convert_st -= 2;
            printfSpaces(lvl_convert_st);
            printf("end of convert while node\n");
#endif
            if (tmp.second)
                bodySt.push(tmp.second);
            if (tmp.first)
                bodySt.push(tmp.first);

            setControlLexNext(inDo);
        }

        if (lastNode->variant() != CONTROL_END)
        {
            pair<SgStatement*, SgStatement*> tmp;
#if TRACE
            printfSpaces(lvl_convert_st);
            printf("convert while node\n");
            lvl_convert_st += 2;
#endif
            (void)convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
            lvl_convert_st -= 2;
            printfSpaces(lvl_convert_st);
            printf("end of convert while node\n");
#endif
            if (tmp.second)
                bodySt.push(tmp.second);
            if (tmp.first)
                bodySt.push(tmp.first);
        }
        else
        {
            pair<SgStatement*, SgStatement*> tmp;
            (void)convertStmt(inDo, tmp, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
            if (tmp.second)
                bodySt.push(tmp.second);
        }

        convertExpr(conditional, conditional);

        if (conditional == NULL)
            conditional = new SgValueExp(1);
        retSt = new SgWhileStmt(conditional, NULL);
        if (cycleName)
        {
            vector<SgLabel*> labs;
            vector<SgStatement*> labsSt;
            createNewLabel(labsSt, labs, cycleName->identifier());

            bodySt.push(labsSt[0]);
            labels_num.insert(labs[0]->thelabel->stateno);
            bodySt.push(new SgContinueStmt());

            bodySt.push(labsSt[1]);
            labels_num.insert(labs[1]->thelabel->stateno);
            bodySt.push(new SgBreakStmt());
        }


        int sizeStack = bodySt.size();
        for (int i = 0; i < sizeStack; ++i)
        {
            retSt->insertStmtAfter(*bodySt.top(), *retSt);
            bodySt.pop();
        }

        needReplace = true;
    }
    else if (st->variant() == SWITCH_NODE)
    {
        SgStatement *tmp = NULL;
        SgStatement *lastNode = st->lastNodeOfStmt();
        stack<SgStatement*> bodySt;

        SgExpression *select = ((SgSwitchStmt*)st)->selector();
        convertExpr(select, select);
        ((SgSwitchStmt*)st)->setSelector(*select);

        //extract default body
        deque<SgStatement*> bodyQueue;
        SgStatement *newIfStmt = NULL;
        tmp = ((SgSwitchStmt*)st)->defOption();
        if (tmp != NULL)
        {
            newIfStmt = new SgIfStmt(*new SgValueExp(0), *new SgStatement(1), 2);

            SgStatement *st = tmp;
            setControlLexNext(tmp);
            st->deleteStmt();
            while (tmp->variant() != CASE_NODE && tmp->variant() != CONTROL_END)
            {
                pair<SgStatement*, SgStatement*> convSt;
#if TRACE
                printfSpaces(lvl_convert_st);
                printf("convert switch node\n");
                lvl_convert_st+=2;
#endif
                (void)convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
                lvl_convert_st -= 2;
                printfSpaces(lvl_convert_st);
                printf("end of convert switch node\n");
#endif
                if (convSt.second)
                    bodyQueue.push_back(convSt.second);
                if (convSt.first)
                    bodyQueue.push_back(convSt.first);
                st = tmp;
                setControlLexNext(tmp);
                st->deleteStmt();

            }
            if (tmp->variant() == CONTROL_END)
            {
                pair<SgStatement*, SgStatement*> convSt;
                (void)convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
                if (convSt.second)
                    bodyQueue.push_back(convSt.second);
            }

            if (!bodyQueue.empty())
            {
                ((SgIfStmt*)newIfStmt)->replaceFalseBody(*bodyQueue.front());
                bodyQueue.pop_front();
                int sizeVector = bodyQueue.size();
                for (int i = 0; i < sizeVector; ++i)
                {
                    ((SgIfStmt*)newIfStmt)->falseBody()->insertStmtAfter(*bodyQueue.back());
                    bodyQueue.pop_back();
                }
            }

        }
        //convert other stmts
        tmp = ((SgSwitchStmt*)st)->caseOption(0);
        if (tmp != NULL)
        {
            if (newIfStmt == NULL)
                newIfStmt = new SgIfStmt(*new SgValueExp(0), *new SgStatement(1), 2);

            pair<SgStatement*, SgStatement*> convSt;
#if TRACE
            printfSpaces(lvl_convert_st);
            printf("convert switch node\n");
            lvl_convert_st+=2;
#endif
            (void)convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
            lvl_convert_st -= 2;
            printfSpaces(lvl_convert_st);
            printf("end of convert switch node\n");
#endif
            if (convSt.second)
                bodySt.push(convSt.second);
            if (convSt.first)
                bodySt.push(convSt.first);
            setControlLexNext(tmp);

            SgExpression * cond = bodySt.top()->expr(0);
            newIfStmt->setExpression(0, *cond);
            bodySt.pop();

            while (tmp != lastNode)
            {
                pair<SgStatement*, SgStatement*> convSt;
#if TRACE
                printfSpaces(lvl_convert_st);
                printf("convert switch node\n");
                lvl_convert_st+=2;
#endif
                (void)convertStmt(tmp, convSt, copyBlock, countOfCopy, lvl + 1, redArraysWithUnknownSize);
#if TRACE
                lvl_convert_st -= 2;
                printfSpaces(lvl_convert_st);
                printf("end of convert switch node\n");
#endif
                if (convSt.second)
                    bodySt.push(convSt.second);
                if (convSt.first)
                    bodySt.push(convSt.first);
                setControlLexNext(tmp);
            }
            int sizeStack = bodySt.size();
            for (int i = 0; i < sizeStack; ++i)
            {
                newIfStmt->insertStmtAfter(*bodySt.top(), *newIfStmt);
                bodySt.pop();
            }
        }

        retSt = newIfStmt;
        needReplace = true;
    }
    else if (st->variant() == CASE_NODE)
    {
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert case node\n");
        lvl_convert_st += 2;
#endif        
        SgExpression *cond = ((SgCaseOptionStmt*)st)->caseRange(0);
        SgExpression *tmpCond = NULL;
        SgExpression *lhs = NULL;
        SgExpression *rhs = NULL;
        SgExpression *select = ((SgSwitchStmt*)(st->controlParent()))->expr(0);
        if (cond->variant() == DDOT)
        {
            lhs = cond->lhs();
            convertExpr(lhs, lhs);
            rhs = cond->rhs();
            convertExpr(rhs, rhs);
            if (rhs == NULL)
                cond = &(*lhs <= *select);
            else if (lhs == NULL)
                cond = &(*select <= *rhs);
            else
                cond = &(*lhs <= *select && *select <= *rhs);
        }
        else
        {
            convertExpr(cond, cond);
            cond = &SgEqOp(*select, *cond);
        }
        for (int i = 1; (tmpCond = ((SgCaseOptionStmt*)st)->caseRange(i)) != 0; ++i)
        {
            if (tmpCond->variant() == DDOT)
            {
                lhs = tmpCond->lhs();
                convertExpr(lhs, lhs);
                rhs = tmpCond->rhs();
                convertExpr(rhs, rhs);
                if (rhs == NULL)
                    tmpCond = &(*lhs <= *select);
                else if (lhs == NULL)
                    tmpCond = &(*select <= *rhs);
                else
                    tmpCond = &(*lhs <= *select && *select <= *rhs);
            }
            else
            {
                convertExpr(tmpCond, tmpCond);
                tmpCond = &SgEqOp(*select, *tmpCond);
            }
            cond = &(*cond || *tmpCond);
        }

        retSt = new SgIfStmt(*cond, *new SgStatement(1), 2);
        retSt->setVariant(ELSEIF_NODE);
#if TRACE
        lvl_convert_st -= 2;
        printfSpaces(lvl_convert_st);
        printf("end of convert case node\n");
#endif
        needReplace = true;
    }
    else if (st->variant() == GOTO_NODE)
    {        
        long lab_num = ((SgGotoStmt*)st)->branchLabel()->thelabel->stateno;
        labels_num.insert(lab_num);
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert goto node\n");
        lvl_convert_st+=2;
#endif
        retSt = &st->copy();
#if TRACE
        lvl_convert_st -= 2;
        printfSpaces(lvl_convert_st);
        printf("end of convert goto node\n");
#endif
        needReplace = false;
    }
    else if (st->variant() == COMGOTO_NODE)
    {
        SgExpression *labList = ((SgComputedGotoStmt*)st)->labelList();
        SgExpression *expr = ((SgComputedGotoStmt*)st)->expr(1);

#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert compute goto node\n");
        lvl_convert_st += 2;
#endif
        convertExpr(expr, expr);
#if TRACE
        lvl_convert_st -= 2;
        printfSpaces(lvl_convert_st);
        printf("end of convert compute goto node\n");
#endif

        int i = 0;
        vector<SgLabel*> labs;
        while (labList)
        {
            SgLabel *lab = ((SgLabelRefExp *)(labList->lhs()))->label();
            SgStatement *labRet = NULL;

            labels_num.insert(lab->thelabel->stateno);
            createNewLabel(labRet, lab);
            labs.push_back(lab);

            labList = labList->rhs();
            i++;
        }
        i--;

        SgIfStmt *if_stat = NULL;
        bool first = true;
        while (i >= 0)
        {
            if (first)
            {
                if_stat = new SgIfStmt(SgEqOp(*expr, *new SgValueExp(i + 1)), *new SgGotoStmt(*labs[i]));
                first = false;
            }
            else
                if_stat = new SgIfStmt(SgEqOp(*expr, *new SgValueExp(i + 1)), *new SgGotoStmt(*labs[i]), *if_stat);
            i--;
        }

        retSt = if_stat;
        needReplace = true;

    }
    else if (st->variant() == PRINT_STAT) // only for SAPFOR
    {
        if (SAPFOR_CONV == 0)
            Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);

        SgInputOutputStmt* outStat = (SgInputOutputStmt*)st;
        SgExpression* lhs = outStat->itemList();
        convertExpr(lhs, lhs);
        
        SgExpression* list = lhs;
        while (list)
        {
            SgExpression* item = list->lhs();
            if (item && item->variant() == STRING_VAL)
            {
                SgValueExp* exp = (SgValueExp*)item;
                string str = exp->stringValue();
                str += "\\n";
                exp->setValue(strdup(str.c_str()));
            }
            list = list->rhs();
        }
        retSt = new SgCExpStmt(*new SgFunctionCallExp(*new SgSymbol(FUNCTION_NAME, "printf"), *lhs));
    }
    else if (st->variant() == PROC_STAT)
    {  
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert call node\n");
        lvl_convert_st += 2;
#endif
        SgExpression *lhs = st->expr(0);
        convertExpr(lhs, lhs);

        if (lhs == NULL || SAPFOR_CONV)
        {
            if (lhs)
                retSt = new SgCExpStmt(*new SgFunctionCallExp(*st->symbol(), *lhs));
            else
                retSt = new SgCExpStmt(*new SgFunctionCallExp(*st->symbol()));
        }
        else
        {
            if (st->symbol()->identifier() == string("random_number"))
            {
                if (lhs->variant() != EXPR_LIST || lhs->lhs() == NULL || lhs->lhs()->variant() != VAR_REF)
                    Error("Unsupported random_number call", "", 660, first_do_par);
                
                //rand state
                lhs->setRhs(new SgExpression(EXPR_LIST, new SgVarRefExp(*new SgSymbol(VARIABLE_NAME, "__dvmh_rand_state")), NULL));
                addRandStateIfNeeded("__dvmh_rand_state");

                retSt = new SgCExpStmt(*new SgFunctionCallExp(*new SgSymbol(VARIABLE_NAME, "__dvmh_rand"), *lhs));                
            }
            else
            {
                SgStatement* inter = getInterfaceForCall(st->symbol());
                if (inter)
                {
                    //switch arguments by keyword
                    lhs = switchArgumentsByKeyword(st->symbol()->identifier(), lhs, inter);
                    //check ommited arguments
                    //transform fact to formal
                }

                matchPrototype(st->symbol(), lhs, false);
                retSt = new SgCExpStmt(*new SgFunctionCallExp(*st->symbol(), *lhs));
            }
        }
#if TRACE
        lvl_convert_st -= 2;
        printfSpaces(lvl_convert_st);
        printf("end of convert call node\n");
#endif
        needReplace = true;
    }
    else if (st->variant() == EXIT_STMT)
    {
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert exit node\n");
        lvl_convert_st += 2;
#endif
        SgSymbol *constrName = ((SgExitStmt*)st)->constructName();
        if (constrName)
        {
            vector<SgLabel*> labs;
            vector<SgStatement*> labsSt;
            createNewLabel(labsSt, labs, constrName->identifier());

            retSt = new SgGotoStmt(*labs[1]);
        }
        else
            retSt = new SgBreakStmt();
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert exit node\n");
#endif
        needReplace = true;
    }
    else if (st->variant() == CYCLE_STMT)
    {
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert cycle node\n");
        lvl_convert_st+=2;
#endif
        SgSymbol *constrName = ((SgCycleStmt*)st)->constructName();
        if (constrName)
        {
            vector<SgLabel*> labs;
            vector<SgStatement*> labsSt;
            createNewLabel(labsSt, labs, constrName->identifier());

            retSt = new SgGotoStmt(*labs[0]);
        }
        else
            retSt = new SgContinueStmt();
#if TRACE
        lvl_convert_st -= 2;
        printfSpaces(lvl_convert_st);
        printf("end of convert cycle node\n");
#endif
        needReplace = true;
    }
    else if (st->variant() == RETURN_STAT)
    {
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert return node\n");
        lvl_convert_st += 2;
#endif
        retSt = new SgReturnStmt();        
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert return node\n");
#endif
        needReplace = true;
    }
    else
    {
        retSt = st;
        if (st->variant() != CONTROL_END && st->variant() != EXPR_STMT_NODE && first_do_par)
        {
            printf("  [STMT ERROR: %s, line %d, user line %d] unsupported variant of node: %s\n", __FILE__, __LINE__, first_do_par->lineNumber(), tag[st->variant()]);
            if (unSupportedVars.size() != 0)
                Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);
        }
    }

    if (lvl > 0)
    {
        if (labSt && retSt)
            retSts = make_pair<SgStatement*, SgStatement*>(&retSt->copy(), &labSt->copy());
        else if (labSt)
            retSts = make_pair<SgStatement*, SgStatement*>(NULL, &labSt->copy());
        else if (retSt)
            retSts = make_pair<SgStatement*, SgStatement*>(&retSt->copy(), NULL);
        else
            retSts = make_pair<SgStatement*, SgStatement*>(NULL, NULL);
    }
    else
    {
        if (retSt)
            retSts = make_pair<SgStatement*, SgStatement*>(&retSt->copy(), NULL);
    }
    return needReplace;
}

void initSupportedVars()
{
    supportedVars.insert(ADD_OP);
    supportedVars.insert(AND_OP);
    supportedVars.insert(NOT_OP);
    supportedVars.insert(DIV_OP);
    supportedVars.insert(EQ_OP);
    supportedVars.insert(EQV_OP);
    supportedVars.insert(EXP_OP);
    supportedVars.insert(GT_OP);
    supportedVars.insert(GTEQL_OP);
    supportedVars.insert(LT_OP);
    supportedVars.insert(LTEQL_OP);
    supportedVars.insert(MINUS_OP);
    supportedVars.insert(MULT_OP);
    supportedVars.insert(NEQV_OP);
    supportedVars.insert(NOTEQL_OP);
    supportedVars.insert(OR_OP);
    supportedVars.insert(SUBT_OP);
    supportedVars.insert(UNARY_ADD_OP);

    supportedVars.insert(BOOL_VAL);
    supportedVars.insert(DOUBLE_VAL);
    supportedVars.insert(FLOAT_VAL);
    supportedVars.insert(INT_VAL);
    supportedVars.insert(COMPLEX_VAL);

    supportedVars.insert(CONST_REF);
    supportedVars.insert(VAR_REF);

    supportedVars.insert(EXPR_LIST);

    supportedVars.insert(FUNC_CALL);
}

void initF2C_FunctionCalls()
{
    handlersOfFunction[string("abs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("and")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("amod")] = FunctionParam("fmod", 2, &createNewFCall);
    handlersOfFunction[string("aimax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("ajmax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("akmax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("aimin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("ajmin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("akmin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("amax1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("amax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("amin1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("amin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("aimag")] = FunctionParam("imag", 1, &createNewFCall);
    handlersOfFunction[string("alog")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("alog10")] = FunctionParam("log10", 1, &createNewFCall);
    handlersOfFunction[string("asin")] = FunctionParam("asin", 1, &createNewFCall);
    handlersOfFunction[string("asind")] = FunctionParam("asin", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("asinh")] = FunctionParam("asinh", 1, &createNewFCall);
    handlersOfFunction[string("acos")] = FunctionParam("acos", 1, &createNewFCall);
    handlersOfFunction[string("acosd")] = FunctionParam("acos", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("acosh")] = FunctionParam("acosh", 1, &createNewFCall);
    handlersOfFunction[string("atan")] = FunctionParam("atan", 1, &createNewFCall);
    handlersOfFunction[string("atand")] = FunctionParam("atan", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("atanh")] = FunctionParam("atanh", 1, &createNewFCall);
    handlersOfFunction[string("atan2")] = FunctionParam("atan2", 2, &createNewFCall);
    handlersOfFunction[string("atan2d")] = FunctionParam("atan2", 0, &__atan2d_handler);
    //intrinsicF.insert(string("aint"));
    //intrinsicF.insert(string("anint"));
    //intrinsicF.insert(string("achar"));
    handlersOfFunction[string("babs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("bbclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("bdim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("biand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("bieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("bior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("bixor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("btest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("bbset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("bbtest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("bbits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("bitest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("bjtest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("bktest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("bessel_j0")] = FunctionParam("j0", 1, &createNewFCall);
    handlersOfFunction[string("bessel_j1")] = FunctionParam("j1", 1, &createNewFCall);
    handlersOfFunction[string("bessel_jn")] = FunctionParam("jn", 2, &createNewFCall);
    handlersOfFunction[string("bessel_y0")] = FunctionParam("y0", 1, &createNewFCall);
    handlersOfFunction[string("bessel_y1")] = FunctionParam("y1", 1, &createNewFCall);
    handlersOfFunction[string("bessel_yn")] = FunctionParam("yn", 2, &createNewFCall);
    handlersOfFunction[string("bmod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("bnot")] = FunctionParam("not", 0, &__not_handler);
    handlersOfFunction[string("bshft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("bshftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("bsign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("cos")] = FunctionParam("cos", 1, &createNewFCall);
    handlersOfFunction[string("ccos")] = FunctionParam("cos", 1, &createNewFCall);
    handlersOfFunction[string("cdcos")] = FunctionParam("cos", 1, &createNewFCall);
    handlersOfFunction[string("cosd")] = FunctionParam("cos", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("cosh")] = FunctionParam("cosh", 1, &createNewFCall);
    handlersOfFunction[string("cotan")] = FunctionParam("tan", 0, &__cotan_handler);
    handlersOfFunction[string("cotand")] = FunctionParam("tan", 0, &__cotand_handler);
    handlersOfFunction[string("cexp")] = FunctionParam("exp", 1, &createNewFCall);
    handlersOfFunction[string("cdexp")] = FunctionParam("exp", 1, &createNewFCall);
    handlersOfFunction[string("conjg")] = FunctionParam("conj", 1, &createNewFCall);
    handlersOfFunction[string("csqrt")] = FunctionParam("sqrt", 1, &createNewFCall);
    handlersOfFunction[string("clog")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("clog10")] = FunctionParam("log10", 1, &createNewFCall);
    handlersOfFunction[string("cdlog")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("cdlog10")] = FunctionParam("log10", 1, &createNewFCall);
    handlersOfFunction[string("cdsqrt")] = FunctionParam("sqrt", 1, &createNewFCall);
    handlersOfFunction[string("csin")] = FunctionParam("sin", 1, &createNewFCall);
    handlersOfFunction[string("ctan")] = FunctionParam("tan", 1, &createNewFCall);
    handlersOfFunction[string("cabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("cdabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("cdsin")] = FunctionParam("sin", 1, &createNewFCall);
    handlersOfFunction[string("cdtan")] = FunctionParam("tan", 1, &createNewFCall);
    handlersOfFunction[string("cmplx")] = FunctionParam("cmplx2", 0, &__cmplx_handler);
    //intrinsicF.insert(string("char"));
    handlersOfFunction[string("dim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("ddim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("dble")] = FunctionParam("double", 1, &createNewFCall);
    handlersOfFunction[string("dfloat")] = FunctionParam("double", 1, &createNewFCall);
    handlersOfFunction[string("dfloti")] = FunctionParam("double", 1, &createNewFCall);
    handlersOfFunction[string("dflotj")] = FunctionParam("double", 1, &createNewFCall);
    handlersOfFunction[string("dflotk")] = FunctionParam("double", 1, &createNewFCall);
    //intrinsicF.insert(string("dint"));
    handlersOfFunction[string("dmax1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("dmin1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("dmod")] = FunctionParam("fmod", 2, &createNewFCall);
    handlersOfFunction[string("dprod")] = FunctionParam("dprod", 2, &createNewFCall);
    handlersOfFunction[string("dreal")] = FunctionParam("real", 1, &createNewFCall);
    handlersOfFunction[string("dsign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("dabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("dsqrt")] = FunctionParam("sqrt", 1, &createNewFCall);
    handlersOfFunction[string("dexp")] = FunctionParam("exp", 1, &createNewFCall);
    handlersOfFunction[string("derf")] = FunctionParam("erf", 1, &createNewFCall);
    handlersOfFunction[string("derfc")] = FunctionParam("erfc", 1, &createNewFCall);
    handlersOfFunction[string("dlog")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("dlog10")] = FunctionParam("log10", 1, &createNewFCall);
    handlersOfFunction[string("dsin")] = FunctionParam("sin", 1, &createNewFCall);
    handlersOfFunction[string("dcos")] = FunctionParam("cos", 1, &createNewFCall);
    handlersOfFunction[string("dcosd")] = FunctionParam("cos", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("dtan")] = FunctionParam("tan", 1, &createNewFCall);
    handlersOfFunction[string("dasin")] = FunctionParam("asin", 1, &createNewFCall);
    handlersOfFunction[string("dasind")] = FunctionParam("asin", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("dasinh")] = FunctionParam("asinh", 1, &createNewFCall);
    handlersOfFunction[string("dacos")] = FunctionParam("acos", 1, &createNewFCall);
    handlersOfFunction[string("dacosd")] = FunctionParam("acos", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("dacosh")] = FunctionParam("acosh", 1, &createNewFCall);
    handlersOfFunction[string("datan")] = FunctionParam("atan", 1, &createNewFCall);
    handlersOfFunction[string("datand")] = FunctionParam("atan", 0, &__arc_sincostan_d_handler);
    handlersOfFunction[string("datanh")] = FunctionParam("atanh", 1, &createNewFCall);
    handlersOfFunction[string("datan2")] = FunctionParam("atan2", 2, &createNewFCall);
    handlersOfFunction[string("datan2d")] = FunctionParam("atan2", 0, &__atan2d_handler);
    handlersOfFunction[string("dsind")] = FunctionParam("sin", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("dsinh")] = FunctionParam("sinh", 1, &createNewFCall);
    handlersOfFunction[string("dcosh")] = FunctionParam("cosh", 1, &createNewFCall);
    handlersOfFunction[string("dcotan")] = FunctionParam("tan", 0, &__cotan_handler);
    handlersOfFunction[string("dcotand")] = FunctionParam("tan", 0, &__cotand_handler);
    handlersOfFunction[string("dshiftl")] = FunctionParam("dshiftl", 3, &createNewFCall);
    handlersOfFunction[string("dshiftr")] = FunctionParam("dshiftr", 3, &createNewFCall);
    handlersOfFunction[string("dtand")] = FunctionParam("tan", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("dtanh")] = FunctionParam("tanh", 1, &createNewFCall);
    //intrinsicF.insert(string("dnint"));
    handlersOfFunction[string("dcmplx")] = FunctionParam("dcmplx2", 0, &__cmplx_handler);
    handlersOfFunction[string("dconjg")] = FunctionParam("conj", 1, &createNewFCall);
    handlersOfFunction[string("dimag")] = FunctionParam("imag", 1, &createNewFCall);
    handlersOfFunction[string("exp")] = FunctionParam("exp", 1, &createNewFCall);
    handlersOfFunction[string("erf")] = FunctionParam("erf", 1, &createNewFCall);
    handlersOfFunction[string("erfc")] = FunctionParam("erfc", 1, &createNewFCall);
    handlersOfFunction[string("erfc_scaled")] = FunctionParam("erfcx", 1, &createNewFCall);
    handlersOfFunction[string("float")] = FunctionParam("float", 1, &createNewFCall);
    handlersOfFunction[string("floati")] = FunctionParam("float", 1, &createNewFCall);
    handlersOfFunction[string("floatj")] = FunctionParam("float", 1, &createNewFCall);
    handlersOfFunction[string("floatk")] = FunctionParam("float", 1, &createNewFCall);
    handlersOfFunction[string("gamma")] = FunctionParam("tgamma", 1, &createNewFCall);
    handlersOfFunction[string("habs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("hbclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("hbits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("hbset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("hdim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("hiand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("hieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("hior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("hixor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("hmod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("hnot")] = FunctionParam("not", 0, &__not_handler);
    handlersOfFunction[string("hshft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("hshftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("hsign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("htest")] = FunctionParam("btest", 2, &createNewFCall);
    handlersOfFunction[string("hypot")] = FunctionParam("hypot", 2, &createNewFCall);
    handlersOfFunction[string("int")] = FunctionParam("int", 1, &createNewFCall);
    handlersOfFunction[string("idint")] = FunctionParam("int", 1, &createNewFCall);
    handlersOfFunction[string("ifix")] = FunctionParam("int", 1, &createNewFCall);
    handlersOfFunction[string("imag")] = FunctionParam("imag", 1, &createNewFCall);
    handlersOfFunction[string("imod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("inot")] = FunctionParam("not", 0, &__not_handler);
    handlersOfFunction[string("idim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("isign")] = FunctionParam("copysign", 2, &createNewFCall);
    //intrinsicF.insert(string("index"));
    handlersOfFunction[string("iabs")] = FunctionParam("abs", 1, &createNewFCall);
    //intrinsicF.insert(string("idnint"));
    //intrinsicF.insert(string("ichar"));
    handlersOfFunction[string("iand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("iiabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("iiand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("iibclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("iibits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("iibset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("iidim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("iieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("iior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("iishft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("iishftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("iisign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("iixor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("ior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("ibset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("ibclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("ibchng")] = FunctionParam("ibchng", 2, &createNewFCall);
    handlersOfFunction[string("ibits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("ieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("ilen")] = FunctionParam("ilen", 1, &createNewFCall);
    handlersOfFunction[string("imax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("imax1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("imin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("imin1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("isha")] = FunctionParam("isha", 2, &createNewFCall);
    handlersOfFunction[string("ishc")] = FunctionParam("ishc", 2, &createNewFCall);
    handlersOfFunction[string("ishft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("ishftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("ishl")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("ixor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("jiabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("jiand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("jibclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("jibits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("jibset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("jidim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("jieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("jior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("jishft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("jishftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("jisign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("jixor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("jmax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("jmax1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("jmin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("jmin1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("jmod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("jnot")] = FunctionParam("not", 0, &__not_handler);
    handlersOfFunction[string("kiabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("kiand")] = FunctionParam("iand", 0, &__iand_handler);
    handlersOfFunction[string("kibclr")] = FunctionParam("ibclr", 2, &createNewFCall);
    handlersOfFunction[string("kibits")] = FunctionParam("ibits", 3, &createNewFCall);
    handlersOfFunction[string("kibset")] = FunctionParam("ibset", 2, &createNewFCall);
    handlersOfFunction[string("kidim")] = FunctionParam("fdim", 2, &createNewFCall);
    handlersOfFunction[string("kieor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("kior")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("kishft")] = FunctionParam("ishft", 2, &createNewFCall);
    handlersOfFunction[string("kishftc")] = FunctionParam("ishftc", 0, &__ishftc_handler);
    handlersOfFunction[string("kisign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("kmax0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("kmax1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("kmin0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("kmin1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("kmod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("knot")] = FunctionParam("not", 0, &__not_handler);
    //intrinsicF.insert(string("len"));
    //intrinsicF.insert(string("lge"));
    //intrinsicF.insert(string("lgt"));
    //intrinsicF.insert(string("lle"));
    //intrinsicF.insert(string("llt"));
    handlersOfFunction[string("log_gamma")] = FunctionParam("lgamma", 1, &createNewFCall);
    handlersOfFunction[string("log")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("log10")] = FunctionParam("log10", 1, &createNewFCall);
    handlersOfFunction[string("lshft")] = FunctionParam("lshft", 2, &createNewFCall);
    handlersOfFunction[string("lshift")] = FunctionParam("lshft", 2, &createNewFCall);
    handlersOfFunction[string("max")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("max0")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("max1")] = FunctionParam("max", 0, &__minmax_handler);
    handlersOfFunction[string("merge_bits")] = FunctionParam("merge_bits", 0, &__merge_bits_handler);
    handlersOfFunction[string("min")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("min0")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("min1")] = FunctionParam("min", 0, &__minmax_handler);
    handlersOfFunction[string("mod")] = FunctionParam("mod", 0, &__mod_handler);
    handlersOfFunction[string("modulo")] = FunctionParam("modulo", 0, &__modulo_handler);
    handlersOfFunction[string("not")] = FunctionParam("not", 0, &__not_handler);
    //intrinsicF.insert(string("nint"));
    handlersOfFunction[string("popcnt")] = FunctionParam("popcnt", 1, &createNewFCall);
    handlersOfFunction[string("poppar")] = FunctionParam("popcnt", 1, &__poppar_handler);
    handlersOfFunction[string("real")] = FunctionParam("real", 1, &createNewFCall);
    handlersOfFunction[string("rshft")] = FunctionParam("rshft", 2, &createNewFCall);
    handlersOfFunction[string("rshift")] = FunctionParam("rshft", 2, &createNewFCall);
    handlersOfFunction[string("or")] = FunctionParam("ior", 0, &__ior_handler);
    handlersOfFunction[string("sign")] = FunctionParam("copysign", 2, &createNewFCall);
    handlersOfFunction[string("sngl")] = FunctionParam("real", 1, &createNewFCall);
    handlersOfFunction[string("sqrt")] = FunctionParam("sqrt", 1, &createNewFCall);
    handlersOfFunction[string("sin")] = FunctionParam("sin", 1, &createNewFCall);
    handlersOfFunction[string("sind")] = FunctionParam("sin", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("sinh")] = FunctionParam("sinh", 1, &createNewFCall);
    handlersOfFunction[string("shifta")] = FunctionParam("shifta", 2, &createNewFCall);
    handlersOfFunction[string("shiftl")] = FunctionParam("lshft", 2, &createNewFCall);
    handlersOfFunction[string("shiftr")] = FunctionParam("shiftr", 2, &createNewFCall);
    handlersOfFunction[string("tan")] = FunctionParam("tan", 1, &createNewFCall);
    handlersOfFunction[string("tand")] = FunctionParam("tan", 0, &__sindcosdtand_handler);
    handlersOfFunction[string("tanh")] = FunctionParam("tanh", 1, &createNewFCall);
    handlersOfFunction[string("trailz")] = FunctionParam("trailz", 1, &createNewFCall);
    handlersOfFunction[string("xor")] = FunctionParam("ieor", 0, &__ieor_handler);
    handlersOfFunction[string("zabs")] = FunctionParam("abs", 1, &createNewFCall);
    handlersOfFunction[string("zcos")] = FunctionParam("cos", 1, &createNewFCall);
    handlersOfFunction[string("zexp")] = FunctionParam("exp", 1, &createNewFCall);
    handlersOfFunction[string("zlog")] = FunctionParam("log", 1, &createNewFCall);
    handlersOfFunction[string("zsin")] = FunctionParam("sin", 1, &createNewFCall);
    handlersOfFunction[string("zsqrt")] = FunctionParam("sqrt", 1, &createNewFCall);
    handlersOfFunction[string("ztan")] = FunctionParam("tan", 1, &createNewFCall);
}

static void correctLabelsUse(SgStatement *firstStmt, SgStatement *lastStmt)
{
    if (firstStmt == lastStmt)
        return;

    SgStatement *copyFSt = firstStmt->lexNext();
    SgStatement *toRem = NULL;
    while (copyFSt != lastStmt)
    {
        if (copyFSt->variant() == LABEL_STAT)
        {
            if (labels_num.find(BIF_LABEL_USE(copyFSt->thebif)->stateno) == labels_num.end())
                toRem = copyFSt;
        }
        copyFSt = copyFSt->lexNext();
        if (toRem != NULL)
        {
            toRem->deleteStmt();
            toRem = NULL;
        }
    }
}

SgStatement* Translate_Fortran_To_C(SgStatement *Stmt, bool isSapforConv)
{
#if TRACE
    printf("START: CONVERTION OF BODY ON LINE %d\n", number_of_loop_line);
#endif
    if (isSapforConv)
    {
        SAPFOR_CONV = 1;
        if (handlersOfFunction.size() == 0)
            initF2C_FunctionCalls();
    }

    map<string, int> redArraysWithUnknownSize;
    SgExpression* er = red_list;
    for (reduction_operation_list* rsl = red_struct_list; rsl && er; rsl = rsl->next, er = er->rhs())
        if (rsl->redvar_size < 0)
            redArraysWithUnknownSize[rsl->redvar->identifier()] = RedFuncNumber(er->lhs()->lhs());

    SgStatement *copyFSt = Stmt;
    SgStatement* last = (Stmt == Stmt->lastNodeOfStmt()) ? Stmt->lastNodeOfStmt() : Stmt->lastExecutable();

    vector<stack<SgStatement*> > copyBlock;
    labelsExitCycle.clear();
    autoTfmReplacing.clear();
    labels_num.clear();
    cond_generator = 0;
    unSupportedVars.clear();
    bool needReplace = false;
    pair<SgStatement*, SgStatement*> converted;

#if TRACE
    printfSpaces(lvl_convert_st);
    printf("convert Stmt\n");
    lvl_convert_st += 2;
#endif
    needReplace = convertStmt(copyFSt, converted, copyBlock, 0, 0, redArraysWithUnknownSize);
#if TRACE
    lvl_convert_st-=2;
    printfSpaces(lvl_convert_st);
    printf("end of convert Stmt\n");
#endif

    if (needReplace && !isSapforConv)
    {       
        char *comm = copyFSt->comments();
        if (comm)
            converted.first->addComment(comm);

        if (converted.first)
            copyFSt->insertStmtBefore(*converted.first, *copyFSt->controlParent());

        copyFSt->deleteStmt();
    }
        
    if (first_do_par)
    {
        for (set<int>::iterator i = unSupportedVars.begin(); i != unSupportedVars.end(); i++)
            printf("  [EXPR ERROR: %s, line %d, %d] unsupported variant of node: %s\n", __FILE__, __LINE__, first_do_par->lineNumber(), tag[*i]);
        if (unSupportedVars.size() != 0)
            Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);
    }

    correctLabelsUse(Stmt, last);

#if TRACE
    printf("END: CONVERTION OF BODY ON LINE %d\n", number_of_loop_line);
#endif

    return converted.first;
}


void Translate_Fortran_To_C(SgStatement *firstStmt, SgStatement *lastStmt, vector<stack<SgStatement*> > &copyBlock, int countOfCopy)
{
#if TRACE
    printf("START: CONVERTION OF BODY ON LINE %d\n", number_of_loop_line);
    lvl_convert_st += 2;
#endif

    map<string, int> redArraysWithUnknownSize;
    SgExpression* er = red_list;
    for (reduction_operation_list* rsl = red_struct_list; rsl && er; rsl = rsl->next, er = er->rhs())
        if (rsl->redvar_size < 0)
            redArraysWithUnknownSize[rsl->redvar->identifier()] = RedFuncNumber(er->lhs()->lhs());

    SgStatement *copyFSt = firstStmt->lexNext();
    vector<SgStatement*> forRemove;        
    labelsExitCycle.clear();
    autoTfmReplacing.clear();
    labels_num.clear();
    unSupportedVars.clear();
    insertAfter.clear();
    insertBefore.clear();
    replaced.clear();
    cond_generator = 0;
    arrayGenNum = 0;

    if (countOfCopy)
        copyBlock = vector<stack< SgStatement*> >(countOfCopy);

    while (copyFSt != lastStmt)
    {
        bool needReplace = false;
        pair<SgStatement*, SgStatement*> converted;
#if TRACE
        printfSpaces(lvl_convert_st);
        printf("convert Stmt\n");
        lvl_convert_st += 2;
#endif        
        needReplace = convertStmt(copyFSt, converted, copyBlock, countOfCopy, 0, redArraysWithUnknownSize);
#if TRACE
        lvl_convert_st-=2;
        printfSpaces(lvl_convert_st);
        printf("end of convert Stmt\n");
#endif
        if (needReplace)
        {
            if (converted.first)
            {
                char *comm = copyFSt->comments();
                if (comm)
                    converted.first->addComment(comm);
                               
                copyFSt->insertStmtBefore(*converted.first, *copyFSt->controlParent());
                replaced[converted.first] = copyFSt;
                for (int i = 0; i < countOfCopy; ++i)
                    copyBlock[i].push(&converted.first->copy());
            }

            SgStatement *tmp1 = copyFSt;
            forRemove.push_back(tmp1);
            setControlLexNext(copyFSt);
        }
        else
            copyFSt = copyFSt->lexNext();
    }

    for (size_t i = 0; i < forRemove.size(); ++i)
        forRemove[i]->deleteStmt();

    for (set<int>::iterator i = unSupportedVars.begin(); i != unSupportedVars.end(); i++)
        printf("  [EXPR ERROR: %s, line %d, %d] unsupported variant of node: %s\n", __FILE__, __LINE__, first_do_par->lineNumber(), tag[*i]);
    if (unSupportedVars.size() != 0)
        Error("Internal inconsistency in F->C convertation", "", 654, first_do_par);

    correctLabelsUse(firstStmt->lexNext(), lastStmt);

    if (options.isOn(AUTO_TFM))
    {
        SgStatement* copyFSt = firstStmt->lexNext();
        if (insertAfter.size() || insertBefore.size())
        {
            while (copyFSt != lastStmt)
            {
                SgStatement* key = (replaced.find(copyFSt) != replaced.end()) ? replaced[copyFSt] : copyFSt;
                if (insertAfter.find(key) != insertAfter.end())
                {
                    for (int z = 0; z < insertAfter[key].size(); ++z)
                        copyFSt->insertStmtAfter(*insertAfter[key][z]);
                }
                if (insertBefore.find(key) != insertBefore.end())
                {
                    for (int z = 0; z < insertBefore[key].size(); ++z)
                        copyFSt->insertStmtBefore(*insertBefore[key][z]);
                }
                copyFSt = copyFSt->lexNext();
            }
        }
    }
#if TRACE
    lvl_convert_st -= 2;
    printf("END: CONVERTION OF BODY ON LINE %d\n", number_of_loop_line);
#endif
}

void  ChangeSymbolName(SgSymbol *symb)
{ 
    char *name = new  char[strlen(symb->identifier())+2];
    sprintf(name, "_%s", symb->identifier());
    SYMB_IDENT(symb->thesymb) = name;
}

void RenamingNewProcedureVariables(SgSymbol *proc_name)
{
    // replacing new procedure names to avoid conflicts with C language keywords and intrinsic function names
    SgSymbol *sl;
    for(sl = proc_name; sl; sl = sl->next())
        switch(sl->variant())
        {
            case VARIABLE_NAME:
            case CONST_NAME:
            case FIELD_NAME:
            case TYPE_NAME:
            case LABEL_VAR:
            case COMMON_NAME:
            case NAMELIST_NAME:
                ChangeSymbolName(sl);
                break;
            default:
               break; 
        }
}

SgSymbol *hasSameNameAsSource(SgSymbol *symb)
{
    symb_list *sl;
    if (!symb)
        return NULL;
    if (sl=isInSymbListByChar(symb, acc_array_list))
        return sl->symb;
    SgExpression *el;
    if (newVars.size() != 0)
    {
        correctPrivateList(RESTORE);
        newVars.clear();
    }
    for (el = private_list; el; el = el->rhs())
        if (!strcmp(el->lhs()->symbol()->identifier(), symb->identifier()))
            return el->lhs()->symbol();
    if (el=isInUsesListByChar(symb->identifier()))
        return el->lhs()->symbol();
    for (el = dvm_parallel_dir ? dvm_parallel_dir->expr(2) : NULL; el; el = el->rhs())
        if (!strcmp(el->lhs()->symbol()->identifier(), symb->identifier()))
            return el->lhs()->symbol();
    reduction_operation_list *rl;
    for (rl = red_struct_list; rl; rl = rl->next)
    { 
        if(rl->redvar && !strcmp(rl->redvar->identifier(), symb->identifier()))
            return rl->redvar;
        if(rl->locvar && !strcmp(rl->locvar->identifier(), symb->identifier()))
            return rl->locvar;
    }
    return NULL;
}

int sameVariableName(SgSymbol *symb1, SgSymbol *symb2)
{
     if (!symb1 || !symb2 || (symb1->variant() != VARIABLE_NAME && symb1->variant() != CONST_NAME && symb1->variant() != FUNCTION_NAME) || symb2->variant() != VARIABLE_NAME && symb2->variant() != CONST_NAME && symb2->variant() != FUNCTION_NAME)
         return 0;
     if (!strcmp (symb1->identifier(), symb2->identifier()))
         return 1;
     else
         return 0;
}

void replaceSymbolSameNameInExpr(SgExpression *expr, SgSymbol *symb, SgSymbol *s_new)
{
    //SgRecordRefExp *re;
    if (!expr || !symb || !s_new)
       return;
    if (expr->symbol())
        if (sameVariableName(expr->symbol(), symb))
            expr->setSymbol(s_new);
    replaceSymbolSameNameInExpr(expr->lhs(), symb, s_new);
    replaceSymbolSameNameInExpr(expr->rhs(), symb, s_new);
}

void replaceVariableSymbSameNameInStatements(SgStatement *first, SgStatement *last, SgSymbol *symb, SgSymbol *s_new, int replace_flag)
{
    SgStatement *stmt;
    for (stmt=first; stmt; stmt = stmt->lexNext())
    {
         if (sameVariableName (stmt->symbol(), symb))
             stmt->setSymbol(*s_new);
         replaceSymbolSameNameInExpr(stmt->expr(0), symb, s_new);
         replaceSymbolSameNameInExpr(stmt->expr(1), symb, s_new);
         replaceSymbolSameNameInExpr(stmt->expr(2), symb, s_new);
         if (last &&  stmt == last)
             break;
    }
}

void RenamingCudaFunctionVariables(SgStatement *first, SgSymbol *k_symb, int replace_flag)
{   // replacing kernel names to avoid conflicts with C language keywords and intrinsic function names
    SgSymbol *sl;
    for (sl=k_symb->next(); sl; sl=sl->next())
    {
        if (sl->scope() != first || sl->variant() != VARIABLE_NAME)
            continue;

        SgSymbol *s_symb = hasSameNameAsSource(sl);
        if (s_symb)
        {   
            if (replace_flag)
                replaceVariableSymbSameNameInStatements(first,first->lastNodeOfStmt(), s_symb, sl, replace_flag);
            ChangeSymbolName(sl);
        }        
    }    
}
