#include "handler_gen.h"

#include <cstdio>

#ifdef WIN32
#include <io.h>
#endif

#include "pass_ctx.h"
#include "messages.h"
#include "converter.h"

#include <llvm/ADT/StringExtras.h>

namespace cdvmh {

// BlankPragmaHandler

#if CLANG_VERSION_MAJOR > 8
void BlankPragmaHandler::HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer, Token &FirstToken) {
#else
void BlankPragmaHandler::HandlePragma(Preprocessor & PP, PragmaIntroducerKind Introducer, Token & FirstToken) {
#endif
    SourceLocation loc = FirstToken.getLocation();
    loc = comp.getSourceManager().getFileLoc(loc);
    FileID fileID = comp.getSourceManager().getFileID(loc);
    int line = comp.getSourceManager().getLineNumber(fileID, comp.getSourceManager().getFileOffset(loc));
    Token &Tok = FirstToken;
    PP.LexNonComment(Tok);
    checkIntErrN(Tok.isAnyIdentifier(), 914);
    std::string tokStr = Tok.getIdentifierInfo()->getName().str();
    checkIntErrN(tokStr == "handler_stub", 914);
    PragmaHandlerStub *curPragma = new PragmaHandlerStub;
    curPragma->line = line;
    curPragma->minAcross = 0;
    curPragma->maxAcross = 0;
    PP.LexNonComment(Tok);
    while (Tok.isAnyIdentifier() || Tok.is(tok::kw_private)) {
        std::string clauseName = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        checkIntErrN(Tok.is(tok::l_paren), 914);
        PP.LexNonComment(Tok);
        if (clauseName == "dvm_array") {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                curPragma->dvmArrays.insert(tokStr);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "regular_array") {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                curPragma->regArrays.insert(tokStr);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "scalar") {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                curPragma->scalars.insert(tokStr);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "loop_var") {
            while (Tok.isAnyIdentifier()) {
                LoopVarDesc loopVar;
                tokStr = Tok.getIdentifierInfo()->getName().str();
                loopVar.name = tokStr;
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::l_paren), 914);
                PP.LexNonComment(Tok);
                tokStr = PP.getSpelling(Tok);
                checkIntErrN(tokStr == "1" || tokStr == "-1", 914);
                loopVar.stepSign = atoi(tokStr.c_str());
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma), 914);
                PP.LexNonComment(Tok);
                if (Tok.isNot(tok::r_paren)) {
                    tokStr = PP.getSpelling(Tok);
                    checkIntErrN(isNumber(tokStr), 914);
                    loopVar.constStep = tokStr;
                    PP.LexNonComment(Tok);
                }
                checkIntErrN(Tok.is(tok::r_paren), 914);
                curPragma->loopVars.push_back(loopVar);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "reduction") {
            while (Tok.isAnyIdentifier()) {
                ClauseReduction red;
                tokStr = Tok.getIdentifierInfo()->getName().str();
                red.redType = ClauseReduction::guessRedType(tokStr);
                checkIntErrN(!red.redType.empty(), 914);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::l_paren), 914);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.isAnyIdentifier(), 914);
                red.arrayName = Tok.getIdentifierInfo()->getName().str();
                PP.LexNonComment(Tok);
                if (red.isLoc()) {
                    checkIntErrN(Tok.is(tok::comma), 914);
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                    red.locName = Tok.getIdentifierInfo()->getName().str();
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.is(tok::comma), 914);
                    PP.LexNonComment(Tok);
                    tokStr = PP.getSpelling(Tok);
                    checkIntErrN(isNumber(tokStr), 914);
                    red.locSize.strExpr = tokStr;
                    PP.LexNonComment(Tok);
                }
                checkIntErrN(Tok.is(tok::r_paren), 914);
                curPragma->reductions.push_back(red);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "private") {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                curPragma->privates.insert(tokStr);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "weird_rma") {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                curPragma->weirdRmas.push_back(tokStr);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.isAnyIdentifier(), 914);
                }
            }
        } else if (clauseName == "across") {
            tokStr = PP.getSpelling(Tok);
            checkIntErrN(isNumber(tokStr), 914);
            curPragma->minAcross = toNumber(tokStr);
            PP.LexNonComment(Tok);
            checkIntErrN(Tok.is(tok::comma), 914);
            PP.LexNonComment(Tok);
            tokStr = PP.getSpelling(Tok);
            checkIntErrN(isNumber(tokStr), 914);
            curPragma->maxAcross = toNumber(tokStr);
            PP.LexNonComment(Tok);
            checkIntErrN(Tok.is(tok::r_paren), 914);
        } else if (clauseName == "remote_access") {
            while (Tok.isAnyIdentifier()) {
                ClauseBlankRma clause;
                clause.nonConstRank = 0;
                tokStr = Tok.getIdentifierInfo()->getName().str();
                clause.origName = tokStr;
                PP.LexNonComment(Tok);
                while (Tok.is(tok::l_square)) {
                    do {
                      PP.LexNonComment(Tok);
                    } while (!Tok.is(tok::r_square));
                }
                while (!Tok.is(tok::l_paren)) {
                    PP.LexNonComment(Tok);
                }
                PP.LexNonComment(Tok);
                while (Tok.is(tok::l_square)) {
                    PP.LexNonComment(Tok);
                    if (Tok.is(tok::r_square)) {
                        clause.indexExprs.push_back("");
                        ++clause.nonConstRank;
                    } else if (Tok.isAnyIdentifier()) {
                        ++clause.nonConstRank;
                        tokStr = Tok.getIdentifierInfo()->getName().str();
                        clause.indexExprs.push_back(tokStr);
                        PP.LexNonComment(Tok);
                    } else {
                        // Zero constant
                        tokStr = PP.getSpelling(Tok);
                        checkIntErrN(tokStr == "0", 914);
                        clause.indexExprs.push_back(tokStr);
                        PP.LexNonComment(Tok);
                    }
                    checkIntErrN(Tok.is(tok::r_square), 914);
                    PP.LexNonComment(Tok);
                }
                checkIntErrN(Tok.is(tok::comma), 914);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.isAnyIdentifier(), 914);
                tokStr = Tok.getIdentifierInfo()->getName().str();
                checkIntErrN(tokStr == "appearances", 914);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::l_paren), 914);
                PP.LexNonComment(Tok);
                while (!Tok.is(tok::r_paren)) {
                    tokStr = PP.getSpelling(Tok);
                    checkIntErrN(isNumber(tokStr), 914);
                    clause.appearances.push_back(toNumber(tokStr));
                    PP.LexNonComment(Tok);
                    checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                    if (Tok.is(tok::comma))
                        PP.LexNonComment(Tok);
                }
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::r_paren), 914);
                curPragma->rmas.push_back(clause);
                PP.LexNonComment(Tok);
                checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 914);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                }
            }
            checkIntErrN(Tok.is(tok::r_paren), 914);
        } else {
            checkIntErrN(false, 914);
        }
        checkIntErrN(Tok.is(tok::r_paren), 914);
        PP.LexNonComment(Tok);
        checkIntErrN(Tok.is(tok::comma) || Tok.is(tok::eod), 914);
        if (Tok.is(tok::comma))
            PP.LexNonComment(Tok);
    }
    checkIntErrN(Tok.is(tok::eod), 914);
    pragmas[line] = curPragma;
}

// BlankRemoteVisitor

bool BlankRemoteVisitor::VisitFunctionDecl(FunctionDecl *f) {
    // Handler cannnot be in macro.
    if (f->getLocStart().isMacroID())
      return true;
    FileID fileID = srcMgr.getFileID(f->getLocStart());
    SourceLocation incLoc = srcMgr.getIncludeLoc(fileID);
    if (incLoc.isValid())
      return true;
    int pragmaLine = srcMgr.getLineNumber(fileID, srcMgr.getFileOffset(f->getLocStart())) - 1;
    PragmaHandlerStub *curPragma = ph->getPragmaAtLine(pragmaLine);
    std::string funcName = f->getName().str();
    bool isHandler = curPragma != 0;
    if (!isHandler || curPragma->rmas.empty()) {
        return true;
    }
    this->curPragma = curPragma;
    parLoopBodyExprCounter = 0;
    inParLoopBody = false;
    parLoopBodyStmt = 0;
    curRma.itr = curPragma->rmas.end();
    CompoundStmt *body = cast<CompoundStmt>(f->getBody());
    for (Stmt **it = body->body_begin(); it != body->body_end(); it++) {
        CompoundStmt *candidate = llvm::dyn_cast<CompoundStmt>(*it);
        if (candidate) {
            parLoopBodyStmt = candidate;
            break;
        }
    }

    std::set<std::string> prohibitedNames;
    CollectNamesVisitor collectNamesVisitor(comp);
    collectNamesVisitor.TraverseStmt(body);
    prohibitedNames = collectNamesVisitor.getNames();
    for (std::set<std::string>::iterator it = curPragma->dvmArrays.begin(); it != curPragma->dvmArrays.end(); it++)
        prohibitedNames.insert(*it);
    for (std::set<std::string>::iterator it = curPragma->regArrays.begin(); it != curPragma->regArrays.end(); it++)
        prohibitedNames.insert(*it);
    for (std::set<std::string>::iterator it = curPragma->scalars.begin(); it != curPragma->scalars.end(); it++)
        prohibitedNames.insert(*it);

    std::string weirdRmasList;
    for (int i = 0; i < (int)curPragma->rmas.size(); i++) {
        std::string substName = getUniqueName(curPragma->rmas[i].origName + "_rma", &prohibitedNames, &seenMacroNames);
        curPragma->rmas[i].substName = substName;
        prohibitedNames.insert(substName);
        weirdRmasList += ", " + substName;
    }
    trimList(weirdRmasList);

    SourceLocation lineBegLoc = srcMgr.translateLineCol(fileID, pragmaLine, 1);
    const char *lineBeg = srcMgr.getBufferData(fileID).data() + srcMgr.getFileOffset(lineBegLoc);
    const char *lineEnd = strchr(lineBeg, '\n');
    SourceLocation lineEndLoc = lineBegLoc.getLocWithOffset(lineEnd - lineBeg);
    rewr.InsertTextAfter(lineEndLoc, ", weird_rma(" + weirdRmasList + ")");

    // TODO: Need to insert them to formal parameters as well :(
    int numParams = f->getNumParams();
    std::string rmaFormalParams;
    for (int i = 0; i < (int)curPragma->rmas.size(); i++) {
        ClauseBlankRma &clause = curPragma->rmas[i];
        int found = -1;
        for (int j = 0; j < numParams; j++) {
            const ParmVarDecl *pvd = f->getParamDecl(j);
            std::string paramName = pvd->getIdentifier()->getName().str();
            if (clause.origName == paramName) {
                found = j;
                break;
            }
        }
        checkIntErrN(found >= 0, 914);
        const ParmVarDecl *pvd = f->getParamDecl(found);
        VarState varState;
        fillVarState(pvd, false, comp, &varState);
        rmaFormalParams += ", " + varState.baseTypeStr + " " + clause.substName;
        if (clause.nonConstRank == 0) {
            rmaFormalParams += "[DVMH_VARIABLE_ARRAY_SIZE]";
        } else {
            for (int j = 0; j < (int)clause.indexExprs.size(); j++) {
                if (clause.indexExprs[j] != "0")
                    rmaFormalParams += "[DVMH_VARIABLE_ARRAY_SIZE]";
            }
        }
    }
    const ParmVarDecl *pvd = f->getParamDecl(numParams - 1);
    SourceLocation endLoc = pvd->getLocEnd();
    endLoc = Lexer::getLocForEndOfToken(endLoc, 0, srcMgr, comp.getLangOpts());
    rewr.InsertTextAfter(endLoc, rmaFormalParams);

    return true;
}

namespace {
struct Predicat {
    Predicat(int c) : exprCounter(c){};
    bool operator()(const ClauseBlankRma &rma) {
        for (std::vector<int>::const_iterator i = rma.appearances.begin(), ei = rma.appearances.end(); i != ei; ++i)
            if (*i == exprCounter)
                return true;
        return false;
    }
    int exprCounter;
};
}

bool BlankRemoteVisitor::TraverseStmt(Stmt *s) {
    if (s == parLoopBodyStmt) {
        inParLoopBody = true;
        bool res = base::TraverseStmt(s);
        inParLoopBody = false;
        return res;
    }
    if (!s || !inParLoopBody)
        return base::TraverseStmt(s);
    ++parLoopBodyExprCounter;
    if (isa<Expr>(s) && curRma.itr == curPragma->rmas.end()) {
        curRma.itr = std::find_if(curPragma->rmas.begin(), curPragma->rmas.end(), Predicat(parLoopBodyExprCounter));
        if (curRma.itr != curPragma->rmas.end()) {
            curRma.curSubscriptIdx = curRma.itr->indexExprs.size() - 1;
            curRma.replacement.clear();
            if (curRma.curSubscriptIdx != 0 || curRma.itr->indexExprs[curRma.curSubscriptIdx].empty() || curRma.itr->indexExprs[curRma.curSubscriptIdx] != "0") {
                // Traverse subscript expressions only if there are non-constant subscripts.
                if (!base::TraverseStmt(s))
                    return false;
            }
            if (curRma.replacement.empty())
                curRma.replacement.push_back(StringRef("[0]"));
            curRma.replacement.push_back(StringRef(curRma.itr->substName));
            rewr.ReplaceText(s->getSourceRange(), llvm::join(curRma.replacement.rbegin(), curRma.replacement.rend(), ""));
            curRma.itr = curPragma->rmas.end();
            return true;
        }
    }
    return base::TraverseStmt(s);
}

bool BlankRemoteVisitor::VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
    if (inParLoopBody && curRma.itr != curPragma->rmas.end()) {
        assert(curRma.curSubscriptIdx >= 0 && "To many subscripts in remote array access!");
        if (!curRma.itr->indexExprs[curRma.curSubscriptIdx].empty()) {
            if (curRma.itr->indexExprs[curRma.curSubscriptIdx] != "0")
                curRma.replacement.push_back(StringRef("[" + curRma.itr->indexExprs[curRma.curSubscriptIdx] + "]"));
        } else {
            curRma.replacement.push_back(StringRef("[" + convertToString(e->getIdx(), rewr.getSourceMgr(), rewr.getLangOpts()) + "]"));
        }
        --curRma.curSubscriptIdx;
    }
    return true;
}

static inline void removePragma(const PragmaHandlerStub& p, const FileID& f, Rewriter& r) {
    const SourceManager &srcMgr = r.getSourceMgr();
    SourceLocation pragmaLineBeginLoc = srcMgr.translateLineCol(f, p.line, 1);
    const char *lineBeg = srcMgr.getBufferData(f).data() + srcMgr.getFileOffset(pragmaLineBeginLoc);
    const char *lineEnd = strchr(lineBeg, '\n');
    r.RemoveText(pragmaLineBeginLoc, lineEnd - lineBeg + 1);
}

static inline std::set<std::string> extractProhibitedNames(CompilerInstance &comp, const PragmaHandlerStub &curPragma, Stmt *functionBody)  {
    std::set<std::string> prohibitedNames;
    CollectNamesVisitor collectNamesVisitor(comp);
    collectNamesVisitor.TraverseStmt(functionBody);
    prohibitedNames = collectNamesVisitor.getNames();
    // Also add declarations from handler_stub pragma which are only passed to the handler as parameters.
    for (std::set<std::string>::iterator it = curPragma.dvmArrays.begin(); it != curPragma.dvmArrays.end(); it++)
        prohibitedNames.insert(*it);
    for (std::set<std::string>::iterator it = curPragma.regArrays.begin(); it != curPragma.regArrays.end(); it++)
        prohibitedNames.insert(*it);
    for (std::set<std::string>::iterator it = curPragma.scalars.begin(); it != curPragma.scalars.end(); it++)
        prohibitedNames.insert(*it);
    return prohibitedNames;
}

static inline Stmt *findHandlerBody(Stmt *functionBody) {
    CompoundStmt *body = cast<CompoundStmt>(functionBody);
    for (Stmt **it = body->body_begin(); it != body->body_end(); it++)
        if (CompoundStmt *candidate = llvm::dyn_cast<CompoundStmt>(*it))
            return candidate;
    return 0;
}

static inline std::pair<const ParmVarDecl *, unsigned> findParameter(const FunctionDecl &f, const std::string& name) {
    for (unsigned j = 0, numParams = f.getNumParams(); j < numParams; j++) {
        const ParmVarDecl *pvd = f.getParamDecl(j);
        if (pvd->getIdentifier()->getName() == name)
            return std::make_pair(pvd, j);
    }
    checkIntErrN(false, 914);
    return std::pair<const ParmVarDecl *, unsigned>(NULL, 0u);
}


namespace {
/// Helper class to generate handler body.
///
/// Inherit it using CRTP.
/// To customize behavior implement '...Imp' methods in a derived class.
template <class BaseT> class HandlerHelper {
public:
    HandlerHelper(const ConverterOptions &aOpts, CompilerInstance &aComp, Rewriter &aRwr, const HandlerFileContext &aBlankCtx,
                  const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &aCurPragma, const FunctionDecl &aF) :
            opts(aOpts), comp(aComp), rwr(aRwr), srcMgr(aRwr.getSourceMgr()), blankCtx(aBlankCtx), curPragma(aCurPragma), f(aF) {
        isSequentialPart = curPragma.loopVars.empty();
        loopRank = (int)curPragma.loopVars.size();
        isAcross = (isSequentialPart ? false : (curPragma.maxAcross != 0 || curPragma.minAcross != 0));
        indent = blankCtx.getIndentStep();
        fileName = srcMgr.getFilename(f.getLocStart()).str();
        bodyToCompute = findHandlerBody(f.getBody());
        for (DeclContext::decl_iterator i = f.decls_begin(), ei = f.decls_end(); i != ei; ++i)
            if (VarDecl *vd = dyn_cast<VarDecl>(*i))
                localDecls[vd->getName()] = vd;
    }

    /// Generate unique names for variables in a handler.
    void genUniqueNames(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesImp(prohibitedGlobal, prohibitedLocal);
    }

    void genUniqueNamesForArrays(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForArraysImp(prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForDvmArrays(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForDvmArraysImp(prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForRmaArrays(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForRmaArraysImp(prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForScalars(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForScalarsImp(prohibitedGlobal, prohibitedLocal);
    }

    void genUniqueNamesForInternal(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForArray(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForDvmArray(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForDvmArrayImp(varState, prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForRmaArray(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForRmaArrayImp(varState, prohibitedGlobal, prohibitedLocal);
    }
    void genUniqueNamesForScalar(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        static_cast<BaseT *>(this)->genUniqueNamesForScalarImp(varState, prohibitedGlobal, prohibitedLocal);
    }

    /// Prepare parameters of a handler (no default implementation for a specific parameter type).
    void prepareParameters() { static_cast<BaseT *>(this)->prepareParametersImp(); }

    void prepareArrayParameters() { static_cast<BaseT *>(this)->prepareArrayParametersImp(); }
    void prepareDvmArrayParameters() { static_cast<BaseT *>(this)->prepareDvmArrayParametersImp(); }
    void prepareScalarParameters() { static_cast<BaseT *>(this)->prepareScalarParametersImp(); }

    void prepareScalarParameter(const VarState &varState, unsigned idx) { static_cast<BaseT *>(this)->prepareScalarParameterImp(varState, idx); }
    void prepareArrayParameter(const VarState &varState, unsigned idx) { static_cast<BaseT *>(this)->prepareArrayParameterImp(varState, idx); }
    void prepareDvmArrayParameter(const VarState &varState, unsigned idx) { static_cast<BaseT *>(this)->prepareDvmArrayParameterImp(varState, idx); }

    /// Prepare remote access buffers (no default implementation for a single remote buffer).
    void prepareRemotes() { static_cast<BaseT *>(this)->prepareRemotesImp(); }
    void prepareRemote(const VarState &varState, int rmaIdx) { static_cast<BaseT *>(this)->prepareRemoteImp(varState, rmaIdx); }

    /// Generate computation body of a handler.
    std::string genBodyToCompute() { return static_cast<BaseT *>(this)->genBodyToComputeImp(); }

    /// Declare all local variables.
    std::string declareLocals(){ return static_cast<BaseT *>(this)->declareLocalsImp(); }

    std::string declareLoopVars() { return static_cast<BaseT *>(this)->declareLoopVarsImp(); }
    std::string declareReductions() { return static_cast<BaseT *>(this)->declareReductionsImp(); }
    std::string declarePrivates() { return static_cast<BaseT *>(this)->declarePrivatesImp(); }

    std::string declareLocal(StringRef v){ return static_cast<BaseT *>(this)->declareLocalImp(v); }
    std::string declareLocal(const LoopVarDesc &v){ return static_cast<BaseT *>(this)->declareLocalImp(v); }
    std::string declareLocal(const ClauseReduction &v){ return static_cast<BaseT *>(this)->declareLocalImp(v); }

public:
    // Default implementation goes bellow.

    void genUniqueNamesImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        genUniqueNamesForInternal(prohibitedGlobal, prohibitedLocal);
        genUniqueNamesForArrays(prohibitedGlobal, prohibitedLocal);
        genUniqueNamesForDvmArrays(prohibitedGlobal, prohibitedLocal);
        genUniqueNamesForRmaArrays(prohibitedGlobal, prohibitedLocal);
        genUniqueNamesForScalars(prohibitedGlobal, prohibitedLocal);
     }

    void genUniqueNamesForArraysImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        for (std::set<std::string>::iterator i = curPragma.regArrays.begin(), ei = curPragma.regArrays.end(); i != ei; ++i) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *i);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            genUniqueNamesForArray(varState, prohibitedGlobal, prohibitedLocal);
        }
     }

    void genUniqueNamesForDvmArraysImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        for (std::set<std::string>::iterator i = curPragma.dvmArrays.begin(), ei = curPragma.dvmArrays.end(); i != ei; ++i) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *i);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            varState.doDvmArray(0);
            genUniqueNamesForDvmArray(varState, prohibitedGlobal, prohibitedLocal);
        }
     }

    void genUniqueNamesForRmaArraysImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        for (std::vector<std::string>::const_iterator i = curPragma.weirdRmas.begin(), ei = curPragma.weirdRmas.end(); i != ei; ++i) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *i);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            varState.doDvmArray(0);
            genUniqueNamesForRmaArray(varState, prohibitedGlobal, prohibitedLocal);
        }
     }

    void genUniqueNamesForScalarsImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        for (std::set<std::string>::iterator i = curPragma.scalars.begin(), ei = curPragma.scalars.end(); i != ei; ++i) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *i);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            genUniqueNamesForScalar(varState, prohibitedGlobal, prohibitedLocal);
        }
     }

    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        device_num = getUniqueName("device_num", &prohibitedLocal, &prohibitedGlobal);
        loop_ref = getUniqueName("loop_ref", &prohibitedLocal, &prohibitedGlobal);
        pLoopRef = getUniqueName("pLoopRef", &prohibitedLocal, &prohibitedGlobal);
        boundsLow = getUniqueName("boundsLow", &prohibitedLocal, &prohibitedGlobal);
        boundsHigh = getUniqueName("boundsHigh", &prohibitedLocal, &prohibitedGlobal);
        loopSteps = getUniqueName("loopSteps", &prohibitedLocal, &prohibitedGlobal);
    }

    void genUniqueNamesForArrayImp(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        assert(varState.isArray && "Variable must be an array!");
        dvmHeaders[varState.name] = getUniqueName(varState.name + "_hdr", &prohibitedLocal, &prohibitedGlobal);
    }

    void genUniqueNamesForDvmArrayImp(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        assert(varState.isDvmArray && "Variable must be a DVM array!");
        genUniqueNamesForArray(varState, prohibitedGlobal, prohibitedLocal);
    }

    void genUniqueNamesForRmaArrayImp(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        assert(varState.isDvmArray && "Variable must be a DVM array!");
        genUniqueNamesForArray(varState, prohibitedGlobal, prohibitedLocal);
    }

    void genUniqueNamesForScalarImp(const VarState& varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        assert(!varState.isArray && "Variable must be a scalar!");
        scalarPtrs[varState.name] = getUniqueName(varState.name + "_ptr", &prohibitedLocal, &prohibitedGlobal);
     }

    void prepareParametersImp() {
        prepareDvmArrayParameters();
        prepareArrayParameters();
        prepareScalarParameters();
    }

    void prepareArrayParametersImp() {
        for (std::set<std::string>::iterator it = curPragma.regArrays.begin(); it != curPragma.regArrays.end(); it++) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *it);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            prepareArrayParameter(varState, pvd.second);
        }
    }

    void prepareDvmArrayParametersImp() {
       for (std::set<std::string>::iterator it = curPragma.dvmArrays.begin(); it != curPragma.dvmArrays.end(); it++) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *it);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            varState.doDvmArray(0);
            prepareDvmArrayParameter(varState, pvd.second);
        }
    }


    void prepareScalarParametersImp() {
        for (std::set<std::string>::iterator it = curPragma.scalars.begin(); it != curPragma.scalars.end(); it++) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *it);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            prepareScalarParameter(varState, pvd.second);
        }
    }

    void prepareScalarParameterImp(const VarState &varState, unsigned idx) {
        assert(!varState.isArray && "Variable must be a scalar!");
    }
    void prepareArrayParameterImp(const VarState &varState, unsigned idx) {
        assert(varState.isArray && "Variable must be an array!");
    }
    void prepareDvmArrayParameterImp(const VarState &varState, unsigned idx) {
        assert(varState.isDvmArray && "Variable must be a DVM array!");
        prepareArrayParameter(varState, idx);
    }

    void prepareRemotesImp() {
       for (std::vector<std::string>::const_iterator bi = curPragma.weirdRmas.begin(), i = curPragma.weirdRmas.begin(), ei = curPragma.weirdRmas.end(); i != ei; ++i) {
            std::pair<const ParmVarDecl *, unsigned> pvd = findParameter(f, *i);
            VarState varState;
            fillVarState(pvd.first, blankCtx.isCPlusPlus(), comp, &varState);
            varState.doDvmArray(0);
            prepareRemote(varState, std::distance(bi, i));
        }
    }

    void prepareRemoteImp(const VarState &varState, int rmaIdx) {
      assert(varState.isDvmArray && "Variable must be a DVM array!");
    }

    std::string genBodyToComputeImp() {
        std::string bodyText;
        std::string bodyStr = rwr.getRewrittenText(bodyToCompute->getSourceRange());
        unsigned firstCol = srcMgr.getExpansionColumnNumber(bodyToCompute->getLocStart());
        unsigned firstIndentDept = firstCol / tabWidth;
        llvm::StringRef bodyRef = bodyStr;
        std::size_t lastPos = 0;
        while (bodyRef.find('\n', lastPos) != llvm::StringRef::npos) {
            std::size_t nlPos = bodyRef.find('\n', lastPos);
            llvm::StringRef lineRef = bodyRef.substr(lastPos, nlPos - lastPos + 1);
            std::string lineIndent = extractIndent(lineRef.data());
            if (lineIndent.size() > firstIndentDept)
                lineIndent = subtractIndent(lineIndent, firstIndentDept);
            bodyText += indent + lineIndent + (lineRef.trim()).str() + "\n";
            lastPos = nlPos + 1;
            if (lastPos >= bodyRef.size())
                break;
        }
        if (lastPos < bodyRef.size())
            bodyText += indent + bodyRef.substr(lastPos).trim().str() + "\n";
        return bodyText;
    }

    std::string declareLocalsImp() {
        return declareLoopVars() + declareReductions() + declarePrivates();
    }

    std::string declareLoopVarsImp() {
        std::string declText;
        for (std::vector<LoopVarDesc>::const_iterator i = curPragma.loopVars.begin(), ei = curPragma.loopVars.end(); i != ei; ++i)
            if (curPragma.privates.find(i->name) == curPragma.privates.end())
                declText += declareLocal(*i);
        return declText;
    }

    std::string declarePrivatesImp() {
        std::string declText;
        for (std::set<std::string>::const_iterator i = curPragma.privates.begin(), ei = curPragma.privates.end(); i != ei; ++i)
            declText += declareLocal(*i);
        return declText;
    }

    std::string declareReductionsImp() {
        std::string declText;
        for (std::vector<ClauseReduction>::const_iterator i = curPragma.reductions.begin(), ei = curPragma.reductions.end(); i != ei; ++i)
            declText += declareLocal(*i);
        return declText;
    }

    std::string declareLocalImp(const LoopVarDesc &v) {
        return declareLocal(v.name);
    }

    std::string declareLocalImp(const ClauseReduction& v) {
        if (v.isLoc())
            return declareLocal(v.locName) + declareLocal(v.arrayName);
        return declareLocal(v.arrayName);
    }

    std::string declareLocalImp(llvm::StringRef name) {
        llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(name);
        checkIntErrN(vdi != localDecls.end(), 914);
        return indent + rwr.getRewrittenText(vdi->second->getSourceRange()) + ";\n";
    }

protected:
    const ConverterOptions &opts;
    CompilerInstance &comp;
    Rewriter &rwr;
    const SourceManager &srcMgr;
    const HandlerFileContext &blankCtx;
    const PragmaHandlerStub &curPragma;
    const FunctionDecl &f;

    bool isSequentialPart;
    int loopRank;
    bool isAcross;
    std::string indent;
    std::string fileName;

    llvm::StringMap<VarDecl *> localDecls;
    Stmt *bodyToCompute;

    std::string device_num;
    std::string loop_ref;
    std::string pLoopRef;
    std::string boundsLow;
    std::string boundsHigh;
    std::string loopSteps;
    std::map<std::string, std::string> dvmHeaders;
    std::map<std::string, std::string> scalarPtrs;
};
}

namespace {
class HostHandlerHelper: public HandlerHelper<HostHandlerHelper> {
public:
    HostHandlerHelper(const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                      const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                      Dvm0CHelper &aDvm0CHelper, const HostReq &aReq) :
            HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f), dvm0CHelper(aDvm0CHelper), req(aReq), handlerFormalParams(f.getNumParams() + 1 - curPragma.weirdRmas.size()) {
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
    }

    std::string generate() {
        genHeading();
        genPrecompute();
        genCompute();
        genPostcompute();
        std::string handlerTemplateDecl; // TODO: template
        return handlerTemplateDecl + "void " + req.handlerName + "(" + llvm::join(handlerFormalParams.begin(), handlerFormalParams.end(), ", ") + ") {\n" + handlerBody + "}\n";
    }

    void genUniqueNamesForInternalImp(const std::set<std::string> &prohibitedGlobal, const std::set<std::string> &prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        slotCount = getUniqueName("slotCount", &prohibitedLocal, &prohibitedGlobal);
        dependencyMask = getUniqueName("dependencyMask", &prohibitedLocal, &prohibitedGlobal);
        currentThread = getUniqueName("currentThread", &prohibitedLocal, &prohibitedGlobal);
        workingThreads = getUniqueName("workingThreads", &prohibitedLocal, &prohibitedGlobal);
        threadSync = getUniqueName("threadSync", &prohibitedLocal, &prohibitedGlobal);
    }

    void prepareArrayParameterImp(const VarState& varState, unsigned idx) {
        const std::string & hdrName = prepareArrayParameterDecl(varState);
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1] = "DvmType " + hdrName + "[]";
    }

    void prepareRemoteImp(const VarState &varState, int rmaIdx) {
        std::string rmaHdr = dvmHeaders[varState.name];
        handlerBody += indent + "DvmType " + rmaHdr + "[" + toStr(varState.headerArraySize) + "];\n";
        handlerBody += indent + "dvmh_loop_get_remote_buf_C(" + loop_ref + ", " + toStr(rmaIdx + 1) + ", " + rmaHdr + ");\n";
        prepareArrayParameterDecl(varState);
    }

    void prepareScalarParameterImp(const VarState& varState, unsigned idx) {
        const std::string &refName = varState.name;
        const std::string &ptrName = scalarPtrs.find(refName)->second;
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1] = varState.baseTypeStr + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + ptrName;
        handlerBody += indent + varState.baseTypeStr + " " + (blankCtx.isCPlusPlus() ? "&" + std::string(varState.canBeRestrict ? " DVMH_RESTRICT_REF " : "") : "") + varState.name + " = *" + ptrName + ";\n";
    }

private:
    void genHeading() {
        assert(0 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[0] = "DvmType *" + pLoopRef;
        handlerBody += "\n";
        handlerBody += indent + "/* Loop reference and device number */\n";
        handlerBody += indent + "DvmType " + loop_ref + " = *" + pLoopRef + ";\n";
        handlerBody += indent + "DvmType " + device_num + " = dvmh_loop_get_device_num_C(" + loop_ref + ");\n";
        handlerBody += indent + "/* Parameters */\n";
        prepareParameters();
        handlerBody += indent + "/* Remote access buffers */\n";
        prepareRemotes();
        if (!isSequentialPart) {
            handlerBody += indent + "/* Supplementary variables for loop handling */\n";
            handlerBody += indent + "DvmType " + boundsLow + "[" + toStr(loopRank) + "], " + boundsHigh + "[" + toStr(loopRank) + "], " + loopSteps + "["
                    + toStr(loopRank) + "];\n";
            if (req.doOpenMP) {
                handlerBody += indent + "int " + slotCount + ";\n";
                if (isAcross)
                    handlerBody += indent + "DvmType " + dependencyMask + ";\n";
            }
        }
        handlerBody += indent + "/* User variables - loop index variables and other private variables */\n";
        handlerBody += declareLocals();
    }

    void genPrecompute() {
        if (!isSequentialPart) {
            handlerBody += indent + "dvmh_loop_fill_bounds_C(" + loop_ref + ", " + boundsLow + ", " + boundsHigh + ", " + loopSteps + ");\n";
        }
        if (req.doOpenMP) {
            handlerBody += indent + slotCount + " = dvmh_loop_get_slot_count_C(" + loop_ref + ");\n";
            if (isAcross) {
                handlerBody += indent + dependencyMask + " = dvmh_loop_get_dependency_mask_C(" + loop_ref + ");\n";
                handlerBody += "#ifdef _OPENMP\n";
                handlerBody += indent + "int " + threadSync + "[" + slotCount + "];\n";
                handlerBody += "#endif\n";
            }
        }
        if (!isSequentialPart)
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                const ClauseReduction &red = curPragma.reductions[i];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState varState;
                fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                if (red.isLoc()) {
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.locName);
                    VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                    checkIntErrN(vd, 97, red.locName.c_str(), fileName.c_str(), curPragma.line);
                    VarState locVarState;
                    fillVarState(vd, blankCtx.isCPlusPlus(), comp, &locVarState);
                    if (!req.doOpenMP || (opts.useOmpReduction && red.hasOpenMP() && !varState.isArray && !locVarState.isArray)) {
                        handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
                        if (opts.dvmDebugLvl & (dlReadVariables | dlWriteVariables)) {
                            handlerBody += indent + "dvmh_dbg_loop_handler_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " +
                                           (varState.isArray ? "" : "&") + red.arrayName + ", \"" + red.arrayName + "\");\n";
                        }
                        if (req.doOpenMP)
                            ompReds.insert(i);
                    }
                } else {
                    if (!req.doOpenMP || (opts.useOmpReduction && red.hasOpenMP() && !varState.isArray)) {
                        handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", 0);\n";
                        if (opts.dvmDebugLvl & (dlReadVariables | dlWriteVariables)) {
                            handlerBody += indent + "dvmh_dbg_loop_handler_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " +
                                           (varState.isArray ? "" : "&") + red.arrayName + ", \"" + red.arrayName + "\");\n";
                        }
                        if (req.doOpenMP)
                            ompReds.insert(i);
                    }
                }
            }
        handlerBody += "\n";
        if (req.doOpenMP) {
            handlerBody += "#ifdef _OPENMP\n";
            handlerBody += indent + "#pragma omp parallel num_threads(" + slotCount + ")";
            if (curPragma.loopVars.size() > 0) {
                handlerBody += ", private(";
                std::string privateList;
                for (std::vector<LoopVarDesc>::const_iterator i = curPragma.loopVars.begin(), ei = curPragma.loopVars.end(); i != ei; ++i)
                    privateList += ", " + i->name;
                trimList(privateList);
                handlerBody += privateList + ")";
            }
            if (curPragma.privates.size() > 0) {
                handlerBody += ", private(";
                std::string privateList;
                for (std::set<std::string>::iterator i = curPragma.privates.begin(), ei = curPragma.privates.end(); i != ei; ++i)
                    privateList += ", " + *i;
                trimList(privateList);
                handlerBody += privateList + ")";
            }
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                if (ompReds.find(i) != ompReds.end())
                    handlerBody += ", reduction(" + curPragma.reductions[i].toOpenMP() + ":" + curPragma.reductions[i].arrayName + ")";
                else
                    handlerBody += ", private(" + curPragma.reductions[i].arrayName +
                            (curPragma.reductions[i].isLoc() ? ", " + curPragma.reductions[i].locName : "") + ")";
            }
            handlerBody += "\n";
            handlerBody += "#endif\n";
            handlerBody += indent + "{\n";
            indent += blankCtx.getIndentStep();
            if (isAcross) {
                handlerBody += "#ifdef _OPENMP\n";
                handlerBody += indent + "int " + currentThread + " = 0, " + workingThreads + " = " + slotCount + ";\n";
                handlerBody += "#endif\n";
            }
            for (int i = 0; i < (int)curPragma.reductions.size(); i++)
                if (ompReds.find(i) == ompReds.end()) {
                    const ClauseReduction &red = curPragma.reductions[i];
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                    VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                    VarState varState;
                    fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                    if (red.isLoc()) {
                        llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.locName);
                        VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                        VarState locVarState;
                        fillVarState(vd, blankCtx.isCPlusPlus(), comp, &locVarState);
                        handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
                    } else
                        handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", 0);\n";
                }
        }
    }

    void genCompute() {
        int loopVariantCount = (req.doOpenMP && isAcross ? loopRank + 1 : 1);
        for (int parInd = 0; parInd < loopVariantCount; parInd++) {
            if (loopVariantCount > 1) {
                handlerBody += indent;
                if (parInd > 0)
                    handlerBody += "} else ";
                if (parInd < loopRank)
                    handlerBody += "if (((" + dependencyMask + " >> " + toStr(loopRank - parInd - 1) + ") & 1) == 0) ";
                handlerBody += "{\n";
                indent += blankCtx.getIndentStep();
            }
            std::string indentSave = indent;
            for (int i = 0; i < loopRank; i++) {
                if (req.doOpenMP && parInd == i) {
                    handlerBody += "#ifdef _OPENMP\n";
                    handlerBody += indent + "#pragma omp for schedule(runtime), nowait\n";
                    handlerBody += "#endif\n";
                }
                if (req.doOpenMP && parInd == loopRank) {
                    // TODO: implement. What?
                    if (i == 0) {
                        handlerBody += "#ifdef _OPENMP\n";
                        std::string paralIters = "(" + boundsHigh + "[1] - " + boundsLow + "[1]) / " + loopSteps + "[1] + 1";
                        handlerBody += indent + "if (" + paralIters + " < " + workingThreads + ")\n";
                        handlerBody += indent + blankCtx.getIndentStep() + workingThreads + " = " + paralIters + ";\n";
                        handlerBody += indent + currentThread + " = omp_get_thread_num();\n";
                        handlerBody += indent + threadSync + "[" + currentThread + "] = 0;\n";
                        handlerBody += indent + "#pragma omp barrier\n";
                        handlerBody += "#endif\n";
                    } else if (i == 1) {
                        handlerBody += subtractIndent(indent) + "{\n";
                        handlerBody += "#ifdef _OPENMP\n";
                        handlerBody += indent + "if (" + currentThread + " > 0 && " + currentThread + " < " + workingThreads + ") {\n";
                        indent += blankCtx.getIndentStep();
                        handlerBody += indent + "do {\n";
                        handlerBody += indent + blankCtx.getIndentStep() + "#pragma omp flush(" + threadSync + ")\n";
                        handlerBody += indent + "} while (!" + threadSync + "[" + currentThread + " - 1]);\n";
                        handlerBody += indent + threadSync + "[" + currentThread + " - 1] = 0;\n";
                        handlerBody += indent + "#pragma omp flush(" + threadSync + ")\n";
                        indent = subtractIndent(indent);
                        handlerBody += indent + "}\n";
                        handlerBody += indent + "#pragma omp for schedule(static), nowait\n";
                        handlerBody += "#endif\n";
                    }
                }
                handlerBody += indent + "for (" + curPragma.loopVars[i].name + " = " + boundsLow + "[" + toStr(i) + "]; " + curPragma.loopVars[i].name + " " +
                        (curPragma.loopVars[i].stepSign > 0 ? "<=" : ">=") + " " + boundsHigh + "[" + toStr(i) + "]; " + curPragma.loopVars[i].name;
                if (curPragma.loopVars[i].constStep == "1")
                    handlerBody += "++";
                else if (curPragma.loopVars[i].constStep == "-1")
                    handlerBody += "--";
                else if (!curPragma.loopVars[i].constStep.empty())
                    handlerBody += " += (" + curPragma.loopVars[i].constStep + ")";
                else
                    handlerBody += " += " + loopSteps + "[" + toStr(i) + "]";
                handlerBody += ")\n";
                indent += blankCtx.getIndentStep();
            }
            if (!isSequentialPart && opts.dvmDebugLvl > 0) {
                handlerBody += indent + "{ dvmh_dbg_loop_iter_C(" + dvm0CHelper.dvm0c(loopRank);
                for (int i = 0; i < loopRank; i++)
                    handlerBody += ", (DvmType)(&" + curPragma.loopVars[i].name + ")";
                for (int i = 0; i < loopRank; i++)
                    handlerBody += ", " + genRtType(curPragma.loopVars[i].baseTypeStr);
                handlerBody += ");\n";
            }
            if (isSequentialPart)
                indent += blankCtx.getIndentStep();
            indent = subtractIndent(indent);
            handlerBody += genBodyToCompute();
            if (!isSequentialPart && opts.dvmDebugLvl > 0)
                handlerBody += indent + "}\n";
            indent = indentSave;
            if (req.doOpenMP && parInd == loopRank) {
                indent += blankCtx.getIndentStep();
                handlerBody += "#ifdef _OPENMP\n";
                handlerBody += indent + "if (" + currentThread + " < " + workingThreads + " - 1) {\n";
                indent += blankCtx.getIndentStep();
                handlerBody += indent + "do {\n";
                handlerBody += indent + blankCtx.getIndentStep() + "#pragma omp flush(" + threadSync + ")\n";
                handlerBody += indent + "} while (" + threadSync + "[" + currentThread + "]);\n";
                handlerBody += indent + threadSync + "[" + currentThread + "] = 1;\n";
                handlerBody += indent + "#pragma omp flush(" + threadSync + ")\n";
                indent = subtractIndent(indent);
                handlerBody += indent + "}\n";
                handlerBody += "#endif\n";
                indent = subtractIndent(indent);
                handlerBody += indent + "}\n";
            }
            if (loopVariantCount > 1)
                indent = subtractIndent(indent);
        }
        if (loopVariantCount > 1)
            handlerBody += indent + "}\n"; // across variants
    }

    void genPostcompute() {
        if (req.doOpenMP) {
            for (int i = 0; i < (int)curPragma.reductions.size(); i++)
                if (ompReds.find(i) == ompReds.end()) {
                    const ClauseReduction &red = curPragma.reductions[i];
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                    VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                    VarState varState;
                    fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                    if (red.isLoc()) {
                        llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.locName);
                        VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                        VarState locVarState;
                        handlerBody += indent + "dvmh_loop_red_post_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +

                                ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
                    } else
                        handlerBody += indent + "dvmh_loop_red_post_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", 0);\n";
                }
            indent = subtractIndent(indent);
            handlerBody += indent + "}\n"; // omp parallel
        }
        if (!isSequentialPart) {
            if (curPragma.reductions.size() > 0)
                handlerBody += "\n";
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                const ClauseReduction &red = curPragma.reductions[i];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                VarState varState;
                fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                assert(vd); // is consequent of upper part
                if (red.isLoc()) {
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.locName);
                    VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                    VarState locVarState;
                    assert(vd); // is consequent of upper part
                    if (!req.doOpenMP || ompReds.find(i) != ompReds.end())
                        handlerBody += indent + "dvmh_loop_red_post_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
                } else {
                    if (!req.doOpenMP || ompReds.find(i) != ompReds.end())
                        handlerBody += indent + "dvmh_loop_red_post_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                                ", 0);\n";
                }
            }
        }
        if (isSequentialPart && !blankCtx.isCPlusPlus()) {
            bool firstFlag = true;
            for (std::set<std::string>::iterator it = curPragma.scalars.begin(); it != curPragma.scalars.end(); it++) {
                if (firstFlag)
                    handlerBody += "\n";
                firstFlag = false;
                handlerBody += indent + "*" + scalarPtrs.find(*it)->second + " = " + *it + ";\n";
            }
        }
    }

    const std::string & prepareArrayParameterDecl(const VarState& varState) {
        int rank = varState.rank;
        const std::string &refName = varState.name;
        const std::string &hdrName = dvmHeaders.find(refName)->second;
        handlerBody += indent + varState.baseTypeStr + " (*" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + refName + ")";
        std::string castType = varState.baseTypeStr + " (*)";
        for (int j = 2; j <= rank; j++) {
            int hdrIdx = j - 1;
            std::string curSize;
            if (varState.isDvmArray || !varState.constSize[j - 1]) {
                if (j < rank)
                    curSize = hdrName + "[" + toStr(hdrIdx) + "]/" + hdrName + "[" + toStr(hdrIdx + 1) + "]";
                else
                    curSize = hdrName + "[" + toStr(hdrIdx) + "]";
            } else {
                curSize = varState.sizes[j - 1].strExpr;
            }
            handlerBody += "[" + curSize + "]";
            castType += "[" + curSize + "]";
        }
        handlerBody += " = (" + castType + ")dvmh_get_natural_base_C(" + device_num + ", " + hdrName + ");\n";
        return hdrName;
    }

private:
    const HostReq &req;
    Dvm0CHelper &dvm0CHelper;

    std::set<int> ompReds;

    std::string slotCount;
    std::string dependencyMask;
    std::string currentThread;
    std::string workingThreads;
    std::string threadSync;

    std::vector<std::string> handlerFormalParams;
    std::string handlerBody;
};
}

// Blank2HostVisitor

bool Blank2HostVisitor::VisitFunctionDecl(FunctionDecl *f) {
    // Handler cannnot be in macro.
    if (f->getLocStart().isMacroID())
      return true;
    FileID fileID = srcMgr.getFileID(f->getLocStart());
    SourceLocation incLoc = srcMgr.getIncludeLoc(fileID);
    if (incLoc.isValid())
      return true;
    int pragmaLine = srcMgr.getExpansionLineNumber(f->getLocStart()) - 1;
    const PragmaHandlerStub *curPragma = ph->getPragmaAtLine(pragmaLine);
    bool isHandler = curPragma != 0;
    //TODO: remove DVMH_VARIABLE_ARRAY_SIZE
    if (!withHeading && firstHandler && isHandler) {
        firstHandler = false;
        if (curPragma->line > 2)
            rewr.RemoveText(SourceRange(srcMgr.getLocForStartOfFile(fileID), srcMgr.translateLineCol(fileID, curPragma->line, 1).getLocWithOffset(-1)));
    }
    std::map<std::string, HostReq>::const_iterator reqItr = blankCtx.getHostReqMap().find(f->getName().str());
    if (reqItr != blankCtx.getHostReqMap().end()) {
        checkIntErrN(isHandler, 915, f->getName().data());
        if (!isOpenMPIncluded && reqItr->second.doOpenMP) {
            isOpenMPIncluded = true;
            std::string includeOpenMP;
            includeOpenMP += "#ifdef _OPENMP\n";
            includeOpenMP += "#include <omp.h>\n";
            includeOpenMP += "#endif\n";
            rewr.InsertText(srcMgr.getLocForStartOfFile(fileID), includeOpenMP);
        }
        removePragma(*curPragma, fileID, rewr);
        //TODO: process templates in C++ sources
        HostHandlerHelper h(opts, comp, rewr, blankCtx, prohibitedNames, *curPragma, *f, dvm0CHelper, reqItr->second);
        std::string handler = h.generate();
        rewr.ReplaceText(f->getSourceRange(), handler);
    } else if (isHandler) {
        SourceRange sr(srcMgr.translateLineCol(fileID, curPragma->line, 1), f->getLocEnd());
        rewr.RemoveText(sr);
    }
    return true;
}

bool Blank2HostVisitor::TraverseFunctionDecl(FunctionDecl *f) {
    bool res = base::TraverseFunctionDecl(f);
    return res;
}

namespace {
class CudaKernelHandlerHelper : public HandlerHelper<CudaKernelHandlerHelper> {
  public:
    CudaKernelHandlerHelper(const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                            const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                            const CudaReq &aReq, const KernelDesc &aKernelDesc) :
        HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f), req(aReq), kernelDesc(aKernelDesc) {
        isOneThread = isAcross || opts.oneThread;
        isAutoTfm = opts.autoTfm;
        isPrepareDiag = isAcross && isAutoTfm; //TODO: add data to handler_stub?: && (curPragma->acrosses.size() > 1 || curPragma->acrosses[0].getDepCount() > 1);
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
    }

    std::string generate() {
        std::string handlerTemplateDecl; //TODO: implement for templates
        std::string kernelName = kernelDesc.kernelName;
        std::string indexT = kernelDesc.indexT;
        kernelBody += indent + "/* Parameters */\n";
        prepareParameters();
        if (!isSequentialPart) {
            if (!isOneThread) {
                kernelBody += indent + "/* Supplementary variables for loop handling */\n";
                kernelBody += indent + indexT + " " + restBlocks + ", " + curBlocks + ";\n";
            }
        }
        kernelBody += indent + "/* User variables - loop index variables and other private variables */\n";
        kernelBody += declareLocals();
        if (!isSequentialPart) {
            for (int i = 0; i < loopRank; i++) {
                kernelFormalParams += ", " + indexT + " " + boundsLow + "_" + toStr(i + 1) +
                        ", " + indexT + " " + boundsHigh + "_" + toStr(i + 1);
                if (curPragma.loopVars[i].constStep.empty())
                    kernelFormalParams += ", " + indexT + " " + loopSteps + "_" + toStr(i + 1);
                if (!isOneThread && i > 0)
                    kernelFormalParams += ", " + indexT + " " + blocksS + "_" + toStr(i + 1);
            }
        }
        if (!isSequentialPart) {
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                // TODO: Add support for reduction arrays
                const ClauseReduction &red = curPragma.reductions[i];
                std::string epsGrid = redGrid[red.arrayName];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState varState;
                fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                // TODO: a diagnostic must point to the original pragma with reduction
                checkUserErrN(!varState.isArray, fileName.c_str(), curPragma.line, 4417, varState.name.c_str());
                kernelFormalParams += ", " + varState.baseTypeStr + " " + red.arrayName;
                kernelFormalParams += ", " + varState.baseTypeStr + " " + epsGrid + "[]";
                if (red.isLoc()) {
                    std::string locGrid = redGrid[red.locName];
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                    VarDecl *lvd = vdi == localDecls.end() ? 0 : vdi->second;
                    checkIntErrN(lvd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                    VarState locVarState;
                    fillVarState(lvd, blankCtx.isCPlusPlus(), comp, &locVarState);
                    // TODO: a diagnostic must point to the original pragma with reduction
                    checkUserErrN(!locVarState.isArray, fileName.c_str(), curPragma.line, 4417, locVarState.name.c_str());
                    kernelFormalParams += ", " + locVarState.baseTypeStr + " " + red.locName;
                    kernelFormalParams += ", " + locVarState.baseTypeStr + " " + locGrid + "[]";

                }
            }
        }
        if (!isSequentialPart && !isOneThread)
            kernelFormalParams += ", " + indexT + " " + blockOffset;
        trimList(kernelFormalParams);
        // kernelFormalParams is done
        for (int i = 1; i <= loopRank; i++) {
            int idx = i - 1;
            if (!isOneThread) {
                if (i == 1)
                    kernelBody += indent + restBlocks + " = blockIdx.x + " + blockOffset + ";\n";
                else
                    kernelBody += indent + restBlocks + " = " + restBlocks + " - " + curBlocks + " * " + blocksS + "_" + toStr(i) + ";\n";
                if (i < loopRank)
                    kernelBody += indent + curBlocks + " = " + restBlocks + " / " + blocksS + "_" + toStr(i + 1) + ";\n";
                else
                    kernelBody += indent + curBlocks + " = " + restBlocks + ";\n";
                kernelBody += indent + curPragma.loopVars[idx].name + " = " + boundsLow + "_" + toStr(i);
                if (curPragma.loopVars[idx].constStep == "1")
                    kernelBody += " + ";
                else if (curPragma.loopVars[idx].constStep == "-1")
                    kernelBody += " - ";
                else if (!curPragma.loopVars[idx].constStep.empty())
                    kernelBody += " + (" + curPragma.loopVars[idx].constStep + ") * ";
                else
                    kernelBody += " + " + loopSteps + "_" + toStr(i) + " * ";
                if (i == loopRank)
                    kernelBody += "(" + curBlocks + " * blockDim.x + threadIdx.x);\n";
                else if (i == loopRank - 1)
                    kernelBody += "(" + curBlocks + " * blockDim.y + threadIdx.y);\n";
                else if (i == loopRank - 2)
                    kernelBody += "(" + curBlocks + " * blockDim.z + threadIdx.z);\n";
                else
                    kernelBody += curBlocks + ";\n";
                kernelBody += indent + "if (" + curPragma.loopVars[idx].name + (curPragma.loopVars[idx].stepSign > 0 ? " <= " : " >= ") + boundsHigh + "_" + toStr(i) + ")" +
                        (i < loopRank ? " {\n" : "\n");
            } else {
                kernelBody += indent + "for (" + curPragma.loopVars[idx].name + " = " + boundsLow + "_" + toStr(i) + "; " + curPragma.loopVars[idx].name +
                        (curPragma.loopVars[idx].stepSign > 0 ? " <= " : " >= ") + boundsHigh + "_" + toStr(i) + "; " + curPragma.loopVars[idx].name;
                if (curPragma.loopVars[idx].constStep == "1")
                    kernelBody += "++";
                else if (curPragma.loopVars[idx].constStep == "-1")
                    kernelBody += "--";
                else if (!curPragma.loopVars[idx].constStep.empty())
                    kernelBody += " += (" + curPragma.loopVars[idx].constStep + ")";
                else
                    kernelBody += " += " + loopSteps + "_" + toStr(i);
                kernelBody += ")\n";
            }
            indent += blankCtx.getIndentStep();
        }
        if (isSequentialPart)
            indent += blankCtx.getIndentStep();
        kernelBody += subtractIndent(indent) + "{\n";
        kernelBody += indent + "do\n";
        kernelBody += genBodyToCompute();
        kernelBody += indent +  "while(0);\n";
        kernelBody += subtractIndent(indent) + "}\n";
        if (!isOneThread)
            for (int i = loopRank - 1; i >= 1; i--)
                kernelBody += genIndent(i, blankCtx.useTabs()) + "}\n";
        indent = blankCtx.getIndentStep();
        if (!isSequentialPart) {
            if (curPragma.reductions.size() > 0) {
                kernelBody += "\n";
                kernelBody += indent + "/* Write reduction values to global memory */\n";
            }
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                // TODO: Add support for reduction arrays
                const ClauseReduction &red = curPragma.reductions[i];
                std::string index;
                if (!isOneThread)
                    index = "threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z + blockDim.z * (blockIdx.x + " + blockOffset +")))";
                else
                    index = "0";
                kernelBody += indent + redGrid[red.arrayName] + "[" + index + "] = " + red.arrayName + ";\n";
                if (red.isLoc())
                    kernelBody += indent + redGrid[red.locName] + "[" + index + "] = " + red.locName + ";\n";
            }
        }
        std::string kernelText;
        kernelText += handlerTemplateDecl;
        kernelText += "__global__ void " + kernelName + "(" + kernelFormalParams + ") {\n";
        kernelText += kernelBody;
        kernelText += "}\n";
        return kernelText;
    }

    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        blocksS = getUniqueName("blocksS", &prohibitedLocal, &prohibitedGlobal);
        restBlocks = getUniqueName("restBlocks", &prohibitedLocal, &prohibitedGlobal);
        blockOffset = getUniqueName("blockOffset", &prohibitedLocal, &prohibitedGlobal);
        curBlocks = getUniqueName("curBlocks", &prohibitedLocal, &prohibitedGlobal);
        for (std::vector<ClauseReduction>::const_iterator i = curPragma.reductions.begin(), ei = curPragma.reductions.end(); i != ei; ++i) {
            redGrid[i->arrayName] = getUniqueName(i->arrayName + "_grid", &prohibitedLocal, &prohibitedGlobal);
            if (i->isLoc())
                redGrid[i->locName] = getUniqueName(i->locName + "_grid", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void genUniqueNamesForArrayImp(const VarState &varState, const std::set<std::string> &prohibitedGlobal, const std::set<std::string> &prohibitedLocal) {
        HandlerHelper::genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
        dvmCoefs[varState.name].clear();
        for (int j = 0; j < varState.headerArraySize; j++)
            dvmCoefs[varState.name].push_back(getUniqueName(varState.name + "_hdr" + toStr(j), &prohibitedLocal, &prohibitedGlobal));
        if (isPrepareDiag) {
            std::map<std::string, std::string> &m = dvmDiagInfos[varState.name];
            m.clear();
            m["tfmType"] = getUniqueName(varState.name + "_tfmType", &prohibitedLocal, &prohibitedGlobal);
            m["xAxis"] = getUniqueName(varState.name + "_xAxis", &prohibitedLocal, &prohibitedGlobal);
            m["yAxis"] = getUniqueName(varState.name + "_yAxis", &prohibitedLocal, &prohibitedGlobal);
            m["Rx"] = getUniqueName(varState.name + "_Rx", &prohibitedLocal, &prohibitedGlobal);
            m["Ry"] = getUniqueName(varState.name + "_Ry", &prohibitedLocal, &prohibitedGlobal);
            m["xOffset"] = getUniqueName(varState.name + "_xOffset", &prohibitedLocal, &prohibitedGlobal);
            m["yOffset"] = getUniqueName(varState.name + "_yOffset", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void prepareParametersImp() {
        prepareDvmArrayParameters();
        prepareArrayParameters();
        prepareScalarParameters();
        prepareRemotes();
    }

    void prepareArrayParameterImp(const VarState &varState, unsigned idx) {
        // XXX: Not so good solution, maybe
        int rank = varState.rank;
        const std::string &refName = varState.name;
        std::string devBaseName = refName + "_base";
        if (rank > 1) {
            std::string elemT = varState.baseTypeStr; // TODO: Add 'const' where appropriate
            std::string ptrT = elemT + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT" : "");
            kernelFormalParams += ", " + ptrT + " " + devBaseName;
            kernelBody += indent + (isAutoTfm ? (isPrepareDiag ? "DvmhDiagonalizedArrayHelper" : "DvmhPermutatedArrayHelper") : "DvmhArrayHelper") + "<" +
                    toStr(rank) + ", " + elemT + ", " + ptrT + ", " + kernelDesc.indexT + "> " + refName + "(" + devBaseName;
            kernelBody += ", DvmhArrayCoefficients<" + toStr(isAutoTfm ? rank : rank - 1) + ", " + kernelDesc.indexT +">(";
            std::string coefList;
            for (int j = 1; j <= rank; j++) {
                int hdrIdx = j;
                std::string coefName = dvmCoefs[refName][hdrIdx];
                if (j < rank || isAutoTfm) {
                    kernelFormalParams += ", " + kernelDesc.indexT + " " + coefName;
                    coefList += ", " + coefName;
                }
            }
            trimList(coefList);
            kernelBody += coefList + ")";
            if (isPrepareDiag) {
                std::map<std::string, std::string> &m = dvmDiagInfos[refName];
                kernelBody += ", DvmhDiagInfo<" + kernelDesc.indexT + ">(";
                kernelFormalParams += ", int " + m["tfmType"];
                kernelBody += m["tfmType"];
                kernelFormalParams += ", int " + m["xAxis"];
                kernelBody += ", " + m["xAxis"];
                kernelFormalParams += ", int " + m["yAxis"];
                kernelBody += ", " + m["yAxis"];
                kernelFormalParams += ", " + kernelDesc.indexT + " " + m["Rx"];
                kernelBody += ", " + m["Rx"];
                kernelFormalParams += ", " + kernelDesc.indexT + " " + m["Ry"];
                kernelBody += ", " + m["Ry"];
                kernelFormalParams += ", " + kernelDesc.indexT + " " + m["xOffset"];
                kernelBody += ", " + m["xOffset"];
                kernelFormalParams += ", " + kernelDesc.indexT + " " + m["yOffset"];
                kernelBody += ", " + m["yOffset"];
                kernelBody += ")";
            }
            kernelBody += ");\n";
        } else {
            kernelFormalParams += ", " + varState.baseTypeStr + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + refName;
        }
    }

    void prepareScalarParameterImp(const VarState &varState, unsigned idx) {
        // TODO: Add 'use' case for variables
        const std::string &refName = varState.name;
        const std::string &ptrName = scalarPtrs.find(refName)->second;
        kernelFormalParams += ", " + varState.baseTypeStr + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + ptrName;
        kernelBody += indent + varState.baseTypeStr + " &" + (varState.canBeRestrict ? " DVMH_RESTRICT_REF " : "") + refName + " = *" + ptrName + ";\n";
    }

    void prepareRemoteImp(const VarState &varState, int rmaIdx) {
        prepareArrayParameter(varState, rmaIdx);
    }

    std::string declareLocalsImp() {
        return declareLoopVars() + declarePrivates();
    }

private:
    const CudaReq &req;
    const KernelDesc &kernelDesc;

    std::string blocksS;
    std::string restBlocks;
    std::string blockOffset;
    std::string curBlocks;
    std::map<std::string, std::vector<std::string> > dvmCoefs;
    std::map<std::string, std::string> redGrid;
    std::map<std::string, std::map<std::string, std::string> > dvmDiagInfos;

    bool isOneThread;
    bool isAutoTfm;
    bool isPrepareDiag;

    std::string kernelFormalParams;
    std::string kernelBody;
};

class CudaHostHandlerHelper : public HandlerHelper<CudaHostHandlerHelper> {
public:
    CudaHostHandlerHelper( const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                      const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                      const CudaReq &aReq) :
      HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f), req(aReq), prohibitedGlobal(prohibitedGlobal), handlerFormalParams(f.getNumParams() + 1 - curPragma.weirdRmas.size()) {
        isOneThread = isAcross || opts.oneThread;
        isAutoTfm = opts.autoTfm;
        isPrepareDiag = isAcross && isAutoTfm; //TODO: add data to handler_stub?: && (curPragma->acrosses.size() > 1 || curPragma->acrosses[0].getDepCount() > 1);
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
    }

    std::pair<std::string, std::string> generate() {
        std::string cudaInfoText;
        // TODO: process templates, also for HOST handlers
        std::string handlerTemplateSpec, handlerTemplateDecl;

        std::vector<KernelDesc> kernelsAvailable;
        kernelsAvailable.push_back(KernelDesc(req.handlerName, "int"));
        kernelsAvailable.push_back(KernelDesc(req.handlerName, "long long"));

        handlerBody += indent + "DvmType " + tmpVar + ";\n";
        assert(0 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[0] = "DvmType *" + pLoopRef;
        handlerBody += indent + "/* Loop reference and device number */\n";
        handlerBody += indent + "DvmType " + loop_ref + " = *" + pLoopRef + ";\n";
        handlerBody += indent + "DvmType " + device_num + " = dvmh_loop_get_device_num_C(" + loop_ref + ");\n";
        handlerBody += indent + "/* Parameters */\n";
        prepareParameters();
        handlerBody += indent + "/* Remote access buffers */\n";
        prepareRemotes();
        // Handler heading is done
        if (!isSequentialPart) {
            handlerBody += indent + "/* Supplementary variables for loop handling */\n";
            handlerBody += indent + "DvmType " + boundsLow + "[" + toStr(loopRank) + "], " + boundsHigh + "[" + toStr(loopRank) + "], "
                        + loopSteps + "[" + toStr(loopRank) + "];\n";
            if (!isOneThread)
                handlerBody += indent + "DvmType " + blocksS + "[" + toStr(loopRank) + "];\n";
        }
        handlerBody += indent + "DvmType " + restBlocks + ";\n";
        handlerBody += indent + "dim3 " + blocks + "(1, 1, 1), " + threads + (isOneThread ? "(1, 1, 1)" : "(0, 0, 0)") + ";\n";
        handlerBody += indent + "cudaStream_t " + stream + ";\n";
        handlerBody += "\n";
        handlerBody += indent + "/* Choose index type for CUDA kernel */\n";
        handlerBody += indent + "int " + kernelIndexT + " = dvmh_loop_guess_index_type_C(" + loop_ref + ");\n";
        handlerBody += indent + "if (" + kernelIndexT + " == rt_LONG) " + kernelIndexT + " = (sizeof(long) <= sizeof(int) ? rt_INT : rt_LLONG);\n";
        handlerBody += indent + "assert(" + kernelIndexT + " == rt_INT || " + kernelIndexT + " == rt_LLONG);\n";
        handlerBody += "\n";
        handlerBody += indent + "/* Get CUDA configuration parameters */\n";
        if (!isSequentialPart) {
            for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
                std::string regsVar = kernelsAvailable[i].regsVar;
                std::string rtIndexT = kernelsAvailable[i].rtIndexT;
                cudaInfoText += "#ifdef " + toUpper(regsVar) + "\n";
                cudaInfoText += indent + "DvmType " + regsVar + " = " + toUpper(regsVar) + ";\n";
                cudaInfoText += "#else\n";
                cudaInfoText += indent + "DvmType " + regsVar + " = 0;\n";
                cudaInfoText += "#endif\n";
                handlerBody += indent + "extern DvmType " + regsVar + ";\n";
                handlerBody += indent + "if (" + kernelIndexT + " == " + rtIndexT + ") dvmh_loop_cuda_get_config_C(" + loop_ref + ", 0, " + regsVar + ", &" +
                        threads + ", &" + stream + ", 0);\n";
            }
        } else {
            handlerBody += indent + "dvmh_loop_cuda_get_config_C(" + loop_ref + ", 0, 0, &" + threads + ", &" + stream + ", 0);\n";
        }
        if (isOneThread)
            handlerBody += indent + threads + " = dim3(1, 1, 1);\n";
        handlerBody += "\n";
        if (!isSequentialPart) {
            handlerBody += indent + "/* Calculate computation distribution parameters */\n";
            handlerBody += indent + "dvmh_loop_fill_bounds_C(" + loop_ref + ", " + boundsLow + ", " + boundsHigh + ", " + loopSteps + ");\n";
            if (!isOneThread) {
                for (int i = loopRank; i >= 1; i--) {
                    int idx = i - 1;
                    std::string threadsCoord;
                    if (i == loopRank)
                        threadsCoord = threads + ".x";
                    else if (i == loopRank - 1)
                        threadsCoord = threads + ".y";
                    else if (i == loopRank - 2)
                        threadsCoord = threads + ".z";
                    else
                        threadsCoord = "1";
                    std::string idxExpr = "[" + toStr(idx) + "]";
                    handlerBody += indent + blocksS + idxExpr + " = ";
                    if (i < loopRank)
                        handlerBody += blocksS + "[" + toStr(idx + 1) + "] * (";
                    handlerBody += "((" + boundsHigh + idxExpr + " - " + boundsLow + idxExpr + " + " + loopSteps + idxExpr + ") / " + loopSteps + idxExpr +
                            " + (" + threadsCoord + " - 1)) / " + threadsCoord;
                    if (i < loopRank)
                        handlerBody += ")";
                    handlerBody += ";\n";
                }
            }
            for (int i = 0; i < loopRank; i++) {
                kernelActualParams += ", " + boundsLow + "[" + toStr(i) + "], " + boundsHigh + "[" + toStr(i) + "]";
                if (curPragma.loopVars[i].constStep.empty())
                    kernelActualParams += ", " + loopSteps + "[" + toStr(i) + "]";
                if (!isOneThread && i > 0)
                    kernelActualParams += ", " + blocksS + "[" + toStr(i) + "]";
            }
            handlerBody += "\n";
        }
        if (!isSequentialPart) {
            if (curPragma.reductions.size() > 0)
                handlerBody += indent + "/* Reductions-related stuff */\n";
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                // TODO: Add support for reduction arrays
                const ClauseReduction &red = curPragma.reductions[i];
                std::string epsGrid = redGrid[red.arrayName];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState varState;
                fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                // TODO: a diagnostic must point of the original pragma with reduction
                checkUserErrN(!varState.isArray, fileName.c_str(), curPragma.line, 4417, varState.name.c_str());
                handlerBody += indent + varState.baseTypeStr + " " + red.arrayName + ";\n";
                handlerBody += indent + varState.baseTypeStr + " *" + epsGrid + ";\n";

                kernelActualParams += ", " + red.arrayName;
                kernelActualParams += ", " + epsGrid;
                if (red.isLoc()) {
                    std::string locGrid = redGrid[red.locName];
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                    VarDecl *lvd = vdi == localDecls.end() ? 0 : vdi->second;
                    checkIntErrN(lvd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                    VarState locVarState;
                    fillVarState(lvd, blankCtx.isCPlusPlus(), comp, &locVarState);
                    // TODO: a diagnostic must point of the original pragma with reduction
                    checkUserErrN(!locVarState.isArray, fileName.c_str(), curPragma.line, 4417, locVarState.name.c_str());
                    handlerBody += indent + locVarState.baseTypeStr + " " + red.locName + ";\n";
                    handlerBody += indent + locVarState.baseTypeStr + " *" + locGrid + ";\n";

                    kernelActualParams += ", " + red.locName;
                    kernelActualParams += ", " + locGrid;

                    handlerBody += indent + "dvmh_loop_cuda_register_red_C(" + loop_ref + ", " + toStr(i + 1) + ", (void **)&" + epsGrid + ", (void **)&" +
                            locGrid + ");\n";
                    handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                            ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
                } else {
                    handlerBody += indent + "dvmh_loop_cuda_register_red_C(" + loop_ref + ", " + toStr(i + 1) + ", (void **)&" + epsGrid + ", 0);\n";
                    handlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                            ", 0);\n";
                }
                if (!isOneThread)
                    handlerBody += indent + "dvmh_loop_cuda_red_prepare_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (blocksS + "[0] * " + threads + ".x * "
                            + threads + ".y * " + threads + ".z") + ", 0);\n";
                else
                    handlerBody += indent + "dvmh_loop_cuda_red_prepare_C(" + loop_ref + ", " + toStr(i + 1) + ", 1, 0);\n";
            }
            if (curPragma.reductions.size() > 0)
                handlerBody += "\n";
        }
        if (!isSequentialPart && !isOneThread)
                kernelActualParams += ", " + blocksS + "[0] - " + restBlocks;
            trimList(kernelActualParams);
            // kernelFactParams is done
            handlerBody += indent + "/* GPU execution */\n";
            if (!isSequentialPart && !isOneThread)
                handlerBody += indent + restBlocks + " = " + blocksS + "[0];\n";
            else
                handlerBody += indent + restBlocks + " = 1;\n";
            handlerBody += indent + "while (" + restBlocks + " > 0) {\n";
            indent += blankCtx.getIndentStep();
            handlerBody += indent + blocks + ".x = (" + restBlocks + " <= 65535 ? " + restBlocks + " : (" + restBlocks + " <= 65535 * 2 ? " + restBlocks +
                    " / 2 : 65535));\n";
            for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
                std::string kernelName = kernelsAvailable[i].kernelName;
                std::string rtIndexT = kernelsAvailable[i].rtIndexT;
                handlerBody += indent + "if (" + kernelIndexT + " == " + rtIndexT + ") " + kernelName + handlerTemplateSpec + "<<<" + blocks + ", " + threads + ", 0, "
                        + stream + ">>>(" + kernelActualParams + ");\n";
            }
            handlerBody += indent + restBlocks + " -= " + blocks + ".x;\n";
            indent = subtractIndent(indent);
            handlerBody += indent + "}\n";
            if (!isSequentialPart) {
                if (curPragma.reductions.size() > 0)
                    handlerBody += "\n";
                for (int i = 0; i < (int)curPragma.reductions.size(); i++)
                    handlerBody += indent + "dvmh_loop_cuda_red_finish_C(" + loop_ref + ", " + toStr(i + 1) + ");\n";
            }
        std::string handlerText;
        for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
            CudaKernelHandlerHelper h(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f, req, kernelsAvailable[i]);
            handlerText += h.generate() + "\n";
        }
        handlerText += (blankCtx.isCPlusPlus() ? handlerTemplateDecl : "extern \"C\" ") + "void " +
                      req.handlerName + "(" + llvm::join(handlerFormalParams.begin(), handlerFormalParams.end(), ", ") + ") {\n" + handlerBody + "}\n";
        return std::make_pair(handlerText, cudaInfoText);
    }

public:
    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        tmpVar = getUniqueName("tmpVar", &prohibitedLocal, &prohibitedGlobal);
        blocksS = getUniqueName("blocksS", &prohibitedLocal, &prohibitedGlobal);
        blocks = getUniqueName("blocks", &prohibitedLocal, &prohibitedGlobal);
        threads = getUniqueName("threads", &prohibitedLocal, &prohibitedGlobal);
        stream = getUniqueName("stream", &prohibitedLocal, &prohibitedGlobal);
        restBlocks = getUniqueName("restBlocks", &prohibitedLocal, &prohibitedGlobal);
        kernelIndexT = getUniqueName("kernelIndexT", &prohibitedLocal, &prohibitedGlobal);
        for (std::vector<ClauseReduction>::const_iterator i = curPragma.reductions.begin(), ei = curPragma.reductions.end(); i != ei; ++i) {
            redGrid[i->arrayName] = getUniqueName(i->arrayName + "_grid", &prohibitedLocal, &prohibitedGlobal);
            if (i->isLoc())
                redGrid[i->locName] = getUniqueName(i->locName + "_grid", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void genUniqueNamesForArrayImp(const VarState &varState, const std::set<std::string> &prohibitedGlobal, const std::set<std::string> &prohibitedLocal) {
        HandlerHelper::genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
        dvmDevHeaders[varState.name] = getUniqueName(varState.name + "_devHdr", &prohibitedLocal, &prohibitedGlobal);
        dvmCoefs[varState.name].clear();
        for (int j = 0; j < varState.headerArraySize; j++)
            dvmCoefs[varState.name].push_back(getUniqueName(varState.name + "_hdr" + toStr(j), &prohibitedLocal, &prohibitedGlobal));
    }

    void prepareArrayParameterImp(const VarState &varState, unsigned idx) {
        std::string hdrName = prepareArrayParameterDecl(varState);
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1] = "DvmType " + hdrName + "[]";
    }

    void prepareScalarParameterImp(const VarState &varState, unsigned idx) {
        // TODO: Add 'use' case for variables
        const std::string &refName = varState.name;
        const std::string &ptrName = scalarPtrs.find(refName)->second;
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1]= varState.baseTypeStr + " *" + ptrName;
        handlerBody += indent + varState.baseTypeStr + " *" + refName;
        handlerBody += " = (" + varState.baseTypeStr + " *)dvmh_get_device_addr_C(" + device_num + ", " + ptrName + ");\n";
        kernelActualParams += ", " + refName;
    }

    void prepareRemoteImp(const VarState &varState, int rmaIdx) {
        std::string rmaHdr = dvmHeaders[varState.name];
        handlerBody += indent + "DvmType " + rmaHdr + "[" + toStr(varState.headerArraySize) + "];\n";
        handlerBody += indent + "dvmh_loop_get_remote_buf_C(" + loop_ref + ", " + toStr(rmaIdx + 1) + ", " + rmaHdr + ");\n";
        prepareArrayParameterDecl(varState);
    }

private:
    std::string prepareArrayParameterDecl(const VarState &varState) {
        int rank = varState.rank;
        const std::string &refName = varState.name;
        std::string hdrName = dvmHeaders.find(refName)->second;
        std::string devHdrName = dvmDevHeaders.find(refName)->second;
        if (isAutoTfm)
            handlerBody += indent + "dvmh_loop_autotransform_C(" + loop_ref + ", " + hdrName + ");\n";
        handlerBody += indent + varState.baseTypeStr + " *" + refName;
        handlerBody += " = (" + varState.baseTypeStr + " *)dvmh_get_natural_base_C(" + device_num + ", " + hdrName + ");\n";
        handlerBody += indent + "DvmType " + devHdrName + "[" + toStr(varState.headerArraySize + (isPrepareDiag ? 7 : 0)) + "];\n";
        handlerBody += indent + tmpVar + " = dvmh_fill_header_C(" + device_num + ", " + refName + ", " + hdrName + ", " + devHdrName + ", " +
                (isPrepareDiag ? devHdrName + " + " + toStr(varState.headerArraySize) : "0") + ");\n";
        if (!isAutoTfm)
            handlerBody += indent + "assert(" + tmpVar + " == 0);\n";
        else if (!isPrepareDiag)
            handlerBody += indent + "assert(" + tmpVar + " == 0 || " + tmpVar + " == 1);\n";
        if (isPrepareDiag) {
            handlerBody += indent + "if (" + tmpVar + " == 2) " + tmpVar + " += 4 * " + devHdrName + "[" + toStr(varState.headerArraySize + 6) + "];\n";
            handlerBody += indent + devHdrName + "[" + toStr(varState.headerArraySize + 6) + "] = " + tmpVar + ";\n";
        }
        kernelActualParams += ", " + refName;
        if (rank > 1) {
            for (int j = 1; j <= rank; j++) {
                int hdrIdx = j;
                if (j < rank || isAutoTfm)
                    kernelActualParams += ", " + devHdrName + "[" + toStr(hdrIdx) + "]";
            }
            if (isPrepareDiag) {
                int diagIdxs[] = {6, 0, 3, 2, 5, 1, 4};
                for (int j = 0; j < 7; j++)
                    kernelActualParams += ", " + devHdrName + "[" + toStr(varState.headerArraySize + diagIdxs[j]) + "]";
            }
        }
        return hdrName;
    }

    const CudaReq &req;
    const std::set<std::string> &prohibitedGlobal;

    bool isOneThread;
    bool isAutoTfm;
    bool isPrepareDiag;

    std::string tmpVar;
    std::string blocksS;
    std::string blocks;
    std::string threads;
    std::string stream;
    std::string restBlocks;
    std::string kernelIndexT;
    std::map<std::string, std::string> dvmDevHeaders;
    std::map<std::string, std::vector<std::string> > dvmCoefs;
    std::map<std::string, std::string> redGrid;

    std::vector<std::string> handlerFormalParams;
    std::string handlerBody;
    std::string kernelActualParams;
};

static void getDefaultCudaBlock(int &x, int &y, int &z, int loopDep, int loopIndep, bool autoTfm) {
    if (autoTfm) {
        if (loopDep == 0) {
            if (loopIndep == 1) {
                x = 256; y = 1; z = 1;
            } else if (loopIndep == 2) {
                x = 32; y = 14; z = 1;
            } else {
                x = 32; y = 7; z = 2;
            }
        } else if (loopDep == 1) {
            if (loopIndep == 0) {
                x = 1; y = 1; z = 1;
            } else if (loopIndep == 1) {
                x = 256; y = 1; z = 1;
            } else if (loopIndep == 2) {
                x = 32; y = 5; z = 1;
            } else {
                x = 16; y = 8; z = 2;
            }
        } else if (loopDep == 2) {
            if (loopIndep == 0) {
                x = 32; y = 1; z = 1;
            } else if (loopIndep == 1){
                x = 32; y = 4; z = 1;
            } else {
                x = 16; y = 8; z = 2;
            }
        } else if (loopDep >= 3) {
            if (loopIndep == 0) {
                x = 32; y = 5; z = 1;
            } else {
                x = 32; y = 5; z = 2;
            }
        }
    } else {
        if (loopDep == 0) {
            if (loopIndep == 1) {
                x = 256; y = 1; z = 1;
            } else if (loopIndep == 2) {
                x = 32; y = 14; z = 1;
            } else {
                x = 32; y = 7; z = 2;
            }
        } else if (loopDep == 1) {
            if (loopIndep == 0) {
                x = 1; y = 1; z = 1;
            } else if (loopIndep == 1) {
                x = 256; y = 1; z = 1;
            } else if (loopIndep == 2) {
                x = 32; y = 8; z = 1;
            } else {
                x = 16; y = 8; z = 2;
            }
        } else if (loopDep == 2) {
            if (loopIndep == 0) {
                x = 32; y = 1; z = 1;
            } else if (loopIndep == 1) {
                x = 32; y = 4; z = 1;
            } else {
                x = 16; y = 8; z = 2;
            }
        } else if (loopDep >= 3) {
            if (loopIndep == 0) {
                x = 8; y = 4; z = 1;
            } else {
                x = 8; y = 4; z = 2;
            }
        }
    }
}

class AcrossCudaKernelCaseHandlerHelper : public HandlerHelper<AcrossCudaKernelCaseHandlerHelper> {
  public:
    AcrossCudaKernelCaseHandlerHelper(const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                                      const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                                      const CudaReq &aReq, const KernelDesc &aKernelDesc, int aDepNumber) :
        HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f), req(aReq), kernelDesc(aKernelDesc), depNumber(aDepNumber) {
        assert(isAcross && "Current helper is for across kernels only!");
        //isOneThread = isAcross || opts.oneThread;
        isAutoTfm = opts.autoTfm;
        isPrepareDiag = isAcross && isAutoTfm && curPragma.maxAcross > 1;
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
    }

    std::string generate() {
        // TODO: implement templates
        std::string handlerTemplateDecl;
        if (depNumber == 1)
            generateOneDepCase();
        else if (depNumber == 2)
            generateTwoDepCase();
        else
            generateOtherDepCase();
        std::string kernelText = handlerTemplateDecl;
        kernelText += "__global__ void " + kernelDesc.kernelName + "(" + caseKernelFormalParams + ") {\n";
        kernelText += caseKernelBody;
        kernelText += indent + "}\n";
        kernelText += "\n";
        return kernelText;
    }

    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        coords = getUniqueName("coords", &prohibitedLocal, &prohibitedGlobal);
        globalThreadIdx = getUniqueName("globalThreadIdx", &prohibitedLocal, &prohibitedGlobal);
        for (std::vector<ClauseReduction>::const_iterator i = curPragma.reductions.begin(), ei = curPragma.reductions.end(); i != ei; ++i) {
            redGrid[i->arrayName] = getUniqueName(i->arrayName + "_grid", &prohibitedLocal, &prohibitedGlobal);
            if (i->isLoc())
                redGrid[i->locName] = getUniqueName(i->locName + "_grid", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void genUniqueNamesForArrayImp(const VarState &varState, const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
        dvmCoefs[varState.name].clear();
        for (int j = 0; j < varState.headerArraySize; j++)
            dvmCoefs[varState.name].push_back(getUniqueName(varState.name + "_hdr" + toStr(j), &prohibitedLocal, &prohibitedGlobal));
        if (isPrepareDiag) {
            std::map<std::string, std::string> &m = dvmDiagInfos[varState.name];
            m.clear();
            m["tfmType"] = getUniqueName(varState.name + "_tfmType", &prohibitedLocal, &prohibitedGlobal);
            m["xAxis"] = getUniqueName(varState.name + "_xAxis", &prohibitedLocal, &prohibitedGlobal);
            m["yAxis"] = getUniqueName(varState.name + "_yAxis", &prohibitedLocal, &prohibitedGlobal);
            m["Rx"] = getUniqueName(varState.name + "_Rx", &prohibitedLocal, &prohibitedGlobal);
            m["Ry"] = getUniqueName(varState.name + "_Ry", &prohibitedLocal, &prohibitedGlobal);
            m["xOffset"] = getUniqueName(varState.name + "_xOffset", &prohibitedLocal, &prohibitedGlobal);
            m["yOffset"] = getUniqueName(varState.name + "_yOffset", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void prepareArrayParameterImp(const VarState& varState, unsigned idx) {
        // XXX: Not so good solution, maybe
        int rank = varState.rank;
        std::string refName = varState.name;
        std::string devBaseName = refName + "_base";
        if (rank > 1) {
            std::string elemT = varState.baseTypeStr; // TODO: Add 'const' where appropriate
            std::string ptrT = elemT + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT" : "");
            caseKernelFormalParams_arrays += ", " + ptrT + " " + devBaseName;
            caseKernelBody += indent + (isAutoTfm ? (isPrepareDiag ? "DvmhDiagonalizedArrayHelper" : "DvmhPermutatedArrayHelper") : "DvmhArrayHelper") + "<" +
                    toStr(rank) + ", " + elemT + ", " + ptrT + ", " + kernelDesc.indexT + "> " + refName + "(" + devBaseName;
            caseKernelBody += ", DvmhArrayCoefficients<" + toStr(isAutoTfm ? rank : rank - 1) + ", " + kernelDesc.indexT +">(";
            std::string coefList;
            for (int j = 1; j <= rank; j++) {
                int hdrIdx = j;
                std::string coefName = dvmCoefs[refName][hdrIdx];
                if (j < rank || isAutoTfm) {
                    caseKernelFormalParams_arrays += ", " + kernelDesc.indexT + " " + coefName;
                    coefList += ", " + coefName;
                }
            }
            trimList(coefList);
            caseKernelBody += coefList + ")";
            if (isPrepareDiag) {
                std::map<std::string, std::string> &m = dvmDiagInfos[refName];
                caseKernelBody += ", DvmhDiagInfo<" + kernelDesc.indexT + ">(";
                caseKernelFormalParams_arrays += ", int " + m["tfmType"];
                caseKernelBody += m["tfmType"];
                caseKernelFormalParams_arrays += ", int " + m["xAxis"];
                caseKernelBody += ", " + m["xAxis"];
                caseKernelFormalParams_arrays += ", int " + m["yAxis"];
                caseKernelBody += ", " + m["yAxis"];
                caseKernelFormalParams_arrays += ", " + kernelDesc.indexT + " " + m["Rx"];
                caseKernelBody += ", " + m["Rx"];
                caseKernelFormalParams_arrays += ", " + kernelDesc.indexT + " " + m["Ry"];
                caseKernelBody += ", " + m["Ry"];
                caseKernelFormalParams_arrays += ", " + kernelDesc.indexT + " " + m["xOffset"];
                caseKernelBody += ", " + m["xOffset"];
                caseKernelFormalParams_arrays += ", " + kernelDesc.indexT + " " + m["yOffset"];
                caseKernelBody += ", " + m["yOffset"];
                caseKernelBody += ")";
            }
            caseKernelBody += ");\n";
        } else {
            caseKernelFormalParams_arrays += ", " + varState.baseTypeStr + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + refName;
        }
    }

    void prepareScalarParameterImp(const VarState & varState, unsigned idx) {
        // TODO: Add 'use' case for variables
        const std::string &ptrName = scalarPtrs.find(varState.name)->second;
        caseKernelFormalParams_scalars += ", " + varState.baseTypeStr + " *" + (varState.canBeRestrict ? " DVMH_RESTRICT " : "") + ptrName;
        caseKernelBody += indent + varState.baseTypeStr + " &" + (varState.canBeRestrict ? " DVMH_RESTRICT_REF " : "") + varState.name + " = *" + ptrName + ";\n";
    }

    void prepareParametersImp() {
        HandlerHelper::prepareParametersImp();
        caseKernelFormalParams += caseKernelFormalParams_arrays + caseKernelFormalParams_scalars;
        caseKernelBody += "\n";
        prepareReductionParameters();
    }

    std::string declareReductionsImp() { return ""; }

private:
    void generateOneDepCase() {
        assert(depNumber == 1 && "Current method is for across for one dependent dimension only!");
        int n_cuda_dims = std::min(loopRank - depNumber, 3);
        char cuda_letter_first = n_cuda_dims == 3 ? 'z' : n_cuda_dims == 2 ? 'y' : 'x';
        char cuda_letter_second = cuda_letter_first - 1;
        char cuda_letter_third = cuda_letter_first - 2;

        caseKernelBody += indent + "/* Parameters */\n";
        prepareParameters();
        for (int i = depNumber; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " num_elem_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " base_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " step_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " idxs_" + toStr(i);
        trimList(caseKernelFormalParams);
        // Kernel formal parameters is done.
        caseKernelBody += indent + "/* User variables - loop index variables and other private variables */\n";
        caseKernelBody += declareLocals();

        caseKernelBody += indent + "/* Computation parameters */\n";
        caseKernelBody += indent + kernelDesc.indexT + " " + coords + "[" + toStr(loopRank) + "];\n";
        for (int i = 0; i < n_cuda_dims; ++i) {
            char letter = 'x' + i;
            caseKernelBody += indent + kernelDesc.indexT + " id_" + toStr(letter) + " = blockIdx." + toStr(letter) +
                    " * blockDim." + toStr(letter) + " + threadIdx." + toStr(letter) + ";\n";
        }
        caseKernelBody += "\n";

        caseKernelBody += indent + "/* Execute one iteration */\n";
        if (loopRank > 1) {
            caseKernelBody += indent + "if (id_" + toStr(cuda_letter_first) + " < num_elem_1";
            if (n_cuda_dims >= 2)
                caseKernelBody += " && id_" + toStr(cuda_letter_second) + " < num_elem_2";
            if (n_cuda_dims == 3) {
                caseKernelBody += " && id_" + toStr(cuda_letter_third) + " < ";
                for (int i = 3; i < loopRank; ++i)
                    if (i == 3)
                        caseKernelBody += "num_elem_" + toStr(i);
                    else
                        caseKernelBody += " * num_elem_" + toStr(i);
            }
            caseKernelBody += ") {\n";
            indent += blankCtx.getIndentStep();
        }

        caseKernelBody += indent + coords + "[idxs_0] = base_0;\n";
        for (int i = 1; i < loopRank; ++i) {
            std::string product;
            for (int j = i + 1; j < loopRank; ++j)
                if (j == i + 1)
                    product += "num_elem_" + toStr(j);
                else
                    product += " * num_elem_" + toStr(j);
            std::string product_division = product.length() == 0 ? "" : " / (" + product + ")";
            char letter = i > 2 ? cuda_letter_third : i > 1 ? cuda_letter_second : cuda_letter_first;
            std::string expression;
            if (i < 3)
                expression = "id_" + toStr(letter);
            else if (i == 3)
                expression = "id_" + toStr(letter) + product_division;
            else if (i > 3 && i < loopRank - 1)
                expression = "(id_" + toStr(letter) + product_division + ") % num_elem_" + toStr(i);
            else if (i == loopRank - 1)
                expression = "id_" + toStr(letter) + " % num_elem_" + toStr(i);
            caseKernelBody += indent + coords + "[idxs_" + toStr(i) + "] = base_" +
                    toStr(i) + " + (" + expression + ") * step_" + toStr(i) + ";\n";
        }

        for (int i = 0; i < (int)curPragma.loopVars.size(); ++i) {
            std::string varName = curPragma.loopVars[i].name;
            caseKernelBody += indent + varName + " = " + coords + "[" + toStr(i) + "];\n";
        }
        caseKernelBody += "\n";
        caseKernelBody += indent + "do\n";
        caseKernelBody += genBodyToCompute();
        caseKernelBody += indent +  "while(0);\n";
        if (loopRank > depNumber) {
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
        }
        endReduction(cuda_letter_first, cuda_letter_second);
        indent = subtractIndent(indent);
    }

    void generateTwoDepCase() {
        assert(depNumber == 2 && "Current method is for across for two dependent dimension only!");
        int n_cuda_dims = 1 + (loopRank >= depNumber + 1 ? 1 : 0) + (loopRank > depNumber + 1 ? 1 : 0);
        char cuda_letter_first = n_cuda_dims == 3 ? 'z' : n_cuda_dims == 2 ? 'y' : 'x';
        char cuda_letter_second = cuda_letter_first - 1;
        char cuda_letter_third = cuda_letter_first - 2;
        caseKernelBody += indent + "/* Parameters */\n";
        prepareParameters();
        caseKernelFormalParams += ", DvmType num_elem_across";
        for (int i = depNumber; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " num_elem_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " base_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " step_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " idxs_" + toStr(i);
        trimList(caseKernelFormalParams);
        // caseKernelFormalParams is done
        caseKernelBody += indent + "/* User variables - loop index variables and other private variables */\n";
        caseKernelBody += declareLocals();
        caseKernelBody += indent + kernelDesc.indexT + " " + coords + "[" + toStr(loopRank) + "];\n";
        for (int i = 0; i < n_cuda_dims; ++i) {
            char letter = 'x' + i;
            caseKernelBody += indent + kernelDesc.indexT + " id_" + toStr(letter) + " = blockIdx." + toStr(letter) +
                    " * blockDim." + toStr(letter) + " + threadIdx." + toStr(letter) + ";\n";
        }
        caseKernelBody += "\n";
        caseKernelBody += indent + "/* Execute one iteration */\n";
        caseKernelBody += indent + "if (id_" + toStr(cuda_letter_first) + " < num_elem_across";
        if (n_cuda_dims >= 2)
            caseKernelBody += " && id_" + toStr(cuda_letter_second) + " < num_elem_2";
        if (n_cuda_dims == 3) {
            caseKernelBody += " && id_" + toStr(cuda_letter_third) + " < ";
            for (int i = 3; i < loopRank; ++i)
                if (i == 3)
                    caseKernelBody += "num_elem_" + toStr(i);
                else
                    caseKernelBody += " * num_elem_" + toStr(i);
        }
        caseKernelBody += ") {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + coords + "[idxs_0] = base_0 - id_" + toStr(cuda_letter_first) + " * step_0;\n";
        caseKernelBody += indent + coords + "[idxs_1] = base_1 + id_" + toStr(cuda_letter_first) + " * step_1;\n";
        for (int i = 2; i < loopRank; ++i) {
            std::string product;
            for (int j = i + 1; j < loopRank; ++j)
                if (j == i + 1)
                    product += "num_elem_" + toStr(j);
                else
                    product += " * num_elem_" + toStr(j);
            std::string product_division = product.length() == 0 ? "" : " / (" + product + ")";
            char letter = i > 2 ? cuda_letter_third : cuda_letter_second;
            std::string expression;
            if (i < 3) {
                expression = "id_" + toStr(letter);
            } else if (i == 3) {
                expression = "id_" + toStr(letter) + product_division;
            } else if (i > 3 && i < loopRank - 1) {
                expression = "(id_" + toStr(letter) + product_division + ") % num_elem_" + toStr(i);
            } else if (i == loopRank - 1) {
                expression = "id_" + toStr(letter) + " % num_elem_" + toStr(i);
            }

            caseKernelBody += indent + coords + "[idxs_" + toStr(i) + "] = base_" +
                    toStr(i) + " + (" + expression + ") * step_" + toStr(i) + ";\n";
        }
        for (int i = 0; i < (int)curPragma.loopVars.size(); ++i) {
            std::string varName = curPragma.loopVars[i].name;
            caseKernelBody += indent + varName + " = " + coords + "[" + toStr(i) + "];\n";
        }
        caseKernelBody += "\n";
        caseKernelBody += indent + "do\n";
        caseKernelBody += genBodyToCompute();
        caseKernelBody += indent +  "while(0);\n";
        if (loopRank > depNumber) {
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
        }
        endReduction(cuda_letter_first, cuda_letter_second);
        if (loopRank == depNumber) {
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
        }
        indent = subtractIndent(indent);
    }

    void generateOtherDepCase() {
        assert(depNumber > 2 && "Current method is for across for more than two dependent dimension only!");
        int n_cuda_dims = 2 + (loopRank > depNumber ? 1 : 0);
        char cuda_letter_first = 'x';
        char cuda_letter_second = 'y';
        char cuda_letter_third = 'z';
        caseKernelBody += indent + "/* Parameters */\n";
        prepareParameters();
         for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " base_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " step_" + toStr(i);
        caseKernelFormalParams += ", DvmType max_z, DvmType SE";
        caseKernelFormalParams += ", DvmType var1, DvmType var2, DvmType var3";
        caseKernelFormalParams += ", DvmType Emax, DvmType Emin";
        caseKernelFormalParams += ", DvmType min_01";
        caseKernelFormalParams += ", DvmType swap_01";

        for (int i = depNumber; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " num_elem_" + toStr(i);
        for (int i = 0; i < loopRank; ++i)
            caseKernelFormalParams += ", " + kernelDesc.indexT + " idxs_" + toStr(i);
        trimList(caseKernelFormalParams);
        // caseKernelFormalParams is done
        caseKernelBody += indent + "/* User variables - loop index variables and other private variables */\n";
        caseKernelBody += declareLocals();
        caseKernelBody += indent + "/* Computation parameters */\n";
        caseKernelBody += indent + kernelDesc.indexT + " " + coords + "[" + toStr(loopRank) + "];\n";
        for (int i = 0; i < n_cuda_dims; ++i) {
            char letter = 'x' + i;
            caseKernelBody += indent + kernelDesc.indexT + " id_" + toStr(letter) + " = blockIdx." + toStr(letter) +
                    " * blockDim." + toStr(letter) + " + threadIdx." + toStr(letter) + ";\n";
        }
        caseKernelBody += "\n";
        caseKernelBody += indent + "/* Execute one iteration */\n";
        caseKernelBody += indent + "if (id_" + toStr(cuda_letter_second) + " < max_z";
        if (n_cuda_dims == 3) {
            caseKernelBody += " && id_" + toStr(cuda_letter_third) + " < ";
            for (int i = 3; i < loopRank; ++i)
                if (i == 3)
                    caseKernelBody += "num_elem_" + toStr(i);
                else
                    caseKernelBody += " * num_elem_" + toStr(i);
        }
        caseKernelBody += ") {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + "if (id_" + toStr(cuda_letter_second) + " + SE < Emin) {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + curPragma.loopVars[0].name + " = id_" + toStr(cuda_letter_second) + " + SE;\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + "if (id_" + toStr(cuda_letter_second) + " + SE < Emax) {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + curPragma.loopVars[0].name + " = min_01;\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + curPragma.loopVars[0].name + " = 2 * min_01 - SE - id_" + toStr(cuda_letter_second) + " + Emax - Emin - 1;\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "}\n";
        caseKernelBody += indent + "if (id_" + toStr(cuda_letter_first) + " < " + curPragma.loopVars[0].name + ") {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + "if (var3 == 1 && Emin < id_" + toStr(cuda_letter_second) + " + SE) {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + "base_0 = base_0 - step_0 * (SE + id_" + toStr(cuda_letter_second) + " - Emin);\n";
        caseKernelBody += indent + "base_1 = base_1 + step_1 * (SE + id_" + toStr(cuda_letter_second) + " - Emin);\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "}\n";
        caseKernelBody += indent + coords + "[idxs_0] = base_0 + (id_" + toStr(cuda_letter_second) + " * (var1 + var3) - id_" + toStr(cuda_letter_first) + ") * step_0;\n";
        caseKernelBody += indent + coords + "[idxs_1] = base_1 + (id_" + toStr(cuda_letter_second) + " * var2 + id_" + toStr(cuda_letter_first) + ") * step_1;\n";
        caseKernelBody += indent + coords + "[idxs_2] = base_2 - id_" + toStr(cuda_letter_second) + " * step_2;\n";
        for (int i = 3; i < depNumber; ++i)
            caseKernelBody += indent + coords + "[idxs_" + toStr(i) + "] = base_" + toStr(i) + ";\n";
        for (int i = depNumber; i < loopRank; ++i) {
            std::string product;
            for (int j = i + 1; j < loopRank; ++j)
                if (j == i + 1)
                    product += "num_elem_" + toStr(j);
                else
                    product += " * num_elem_" + toStr(j);
            std::string product_division = product.length() == 0 ? "" : " / (" + product + ")";
            char letter = cuda_letter_third;
            std::string expression;
            if (i == 3)
                expression = "id_" + toStr(letter) + product_division;
            else if (i > 3 && i < loopRank - 1)
                expression = "(id_" + toStr(letter) + product_division + ") % num_elem_" + toStr(i);
            else if (i == loopRank - 1)
                expression = "id_" + toStr(letter) + " % num_elem_" + toStr(i);
            caseKernelBody += indent + coords + "[idxs_" + toStr(i) + "] = base_" +
                    toStr(i) + " + (" + expression + ") * step_" + toStr(i) + ";\n";
        }
        caseKernelBody += indent + "if (swap_01 * var3) {\n";
        indent += blankCtx.getIndentStep();
        caseKernelBody += indent + "var3 = " + coords + "[idxs_1];\n";
        caseKernelBody += indent + coords + "[idxs_1] = " + coords + "[idxs_0];\n";
        caseKernelBody += indent + coords + "[idxs_0] = var3;\n";
        indent = subtractIndent(indent);
        caseKernelBody += indent + "}\n";
        for (int i = 0; i < (int)curPragma.loopVars.size(); ++i) {
            std::string varName = curPragma.loopVars[i].name;
            caseKernelBody += indent + varName + " = " + coords + "[" + toStr(i) + "];\n";
        }
        caseKernelBody += "\n";
        caseKernelBody += indent + "do\n";
        caseKernelBody += genBodyToCompute();
        caseKernelBody += indent +  "while(0);\n";
        if (loopRank > depNumber) {
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
        }
        endReduction(cuda_letter_first, cuda_letter_second);
        if (loopRank == depNumber) {
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
            indent = subtractIndent(indent);
            caseKernelBody += indent + "}\n";
        }
        indent = subtractIndent(indent);
    }

    void prepareReductionParameters() {
        for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
            // TODO: Add support for reduction arrays
            const ClauseReduction &red = curPragma.reductions[i];
            std::string epsGrid = redGrid[red.arrayName];
            llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
            VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
            checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
            VarState varState;
            fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
            // TODO: a diagnostic must point to the original pragma with reduction
            checkUserErrN(!varState.isArray, fileName.c_str(), curPragma.line, 4417, varState.name.c_str());
            caseKernelFormalParams += ", " + varState.baseTypeStr + " " + red.arrayName;
            caseKernelFormalParams += ", " + varState.baseTypeStr + " " + epsGrid + "[]";
            if (red.isLoc()) {
                std::string locGrid = redGrid[red.locName];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *lvd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(lvd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState locVarState;
                fillVarState(lvd, blankCtx.isCPlusPlus(), comp, &locVarState);
                // TODO: a diagnostic must point of the original pragma with reduction
                checkUserErrN(!locVarState.isArray, fileName.c_str(), curPragma.line, 4417, locVarState.name.c_str());
                caseKernelFormalParams += ", " + locVarState.baseTypeStr + " " + red.locName;
                caseKernelFormalParams += ", " + locVarState.baseTypeStr + " " + locGrid + "[]";
            }
        }
    }

    void endReduction(char cuda_letter_first, char cuda_letter_second) {
        if (curPragma.reductions.size() > 0) {
            caseKernelBody += "\n";
            caseKernelBody += indent + "/* Reduction */\n";
            for (int i = 0; i < (int)curPragma.reductions.size(); i++) {
                // TODO: Add support for reduction arrays
                const ClauseReduction &red = curPragma.reductions[i];
                std::string redType = red.redType;
                std::string epsGrid = redGrid[red.arrayName];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState varState;
                fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
                // TODO: a diagnostic must point to the original pragma with reduction
                checkUserErrN(!varState.isArray, fileName.c_str(), curPragma.line, 4417, varState.name.c_str());
                std::pair<std::string, bool> op = red.toCUDA();
                assert(!op.first.empty() && "Unsupported type of a reduction operation!");
                if (red.isLoc()) {
                    std::string locGrid = redGrid[red.locName];
                    llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                    VarDecl *lvd = vdi == localDecls.end() ? 0 : vdi->second;
                    checkIntErrN(lvd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                    VarState locVarState;
                    fillVarState(lvd, blankCtx.isCPlusPlus(), comp, &locVarState);
                    // TODO: a diagnostic must point of the original pragma with reduction
                    checkUserErrN(!locVarState.isArray, fileName.c_str(), curPragma.line, 4417, locVarState.name.c_str());
                    if (loopRank > depNumber) {
                        caseKernelBody += indent + kernelDesc.indexT + " " + globalThreadIdx + " = " +
                            "threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + "
                            "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y * blockDim.z;\n";
                        caseKernelBody += indent + red.arrayName + " = "
                                        + red.toCUDABlock() + "<" + varState.baseTypeStr + "," + locVarState.baseTypeStr + ", " + toStr(varState.rank) + ">"
                                        +"(" + red.arrayName + ", &" + red.locName + ");\n";
                        caseKernelBody += indent + "if (" + globalThreadIdx + " % warpSize == 0) {\n";
                        indent += blankCtx.getIndentStep();
                        caseKernelBody += indent + "if (" + red.arrayName + (red.redType == "rf_MINLOC" ? " < " : " > ") + epsGrid + "[" + globalThreadIdx + " / warpSize]) {\n";
                        indent += blankCtx.getIndentStep();
                        caseKernelBody += indent + epsGrid + "[" + globalThreadIdx + " / warpSize] = " + red.arrayName + ";\n";
                        caseKernelBody += indent + locGrid + "[" + globalThreadIdx + " / warpSize] = " + red.locName + ";\n";
                        indent = subtractIndent(indent);
                        caseKernelBody += "}\n";
                        indent = subtractIndent(indent);
                        caseKernelBody += "}\n";
                    } else {
                        std::string index_expr;
                        if (loopRank == 1)
                            index_expr = "[0]";
                        else if (depNumber > 2)
                          index_expr = "[id_" + toStr(cuda_letter_first) + " + " + "id_" + toStr(cuda_letter_second) + " * Emin" + "]";
                        else
                          index_expr = "[id_" + toStr(cuda_letter_first) + "]";
                        caseKernelBody += indent + "if (" + red.arrayName + (red.redType == "rf_MINLOC" ? " < " : " > ") + epsGrid + "[id_" + cuda_letter_first + "]) {\n";
                        indent += blankCtx.getIndentStep();
                        caseKernelBody += indent + epsGrid + index_expr + " = " + red.arrayName + ";\n";
                        caseKernelBody += indent + locGrid + index_expr + " = " + red.locName + ";\n";
                        indent = subtractIndent(indent);
                        caseKernelBody += "}\n";
                    }
                } else {
                    if (loopRank > depNumber) {
                        caseKernelBody += indent + kernelDesc.indexT + " " + globalThreadIdx + " = " +
                            "threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + "
                            "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * blockDim.x * blockDim.y * blockDim.z;\n";
                        caseKernelBody += indent + red.arrayName + " = " + red.toCUDABlock() + "(" + red.arrayName + ");\n";
                        caseKernelBody += indent + "if (" + globalThreadIdx + " % warpSize == 0) {\n";
                        indent += blankCtx.getIndentStep();
                        if (op.second)
                            caseKernelBody += indent + epsGrid + "[" + globalThreadIdx + " / warpSize] = " + op.first + "(" + epsGrid + "[" + globalThreadIdx + " / warpSize], " + red.arrayName + ");\n";
                        else
                            caseKernelBody += indent + epsGrid + "[" + globalThreadIdx + " / warpSize] = " + epsGrid + "[" + globalThreadIdx + " / warpSize] " + op.first + " " + red.arrayName + ";\n";
                        indent = subtractIndent(indent);
                        caseKernelBody += indent + "}\n";
                    } else {
                        std::string index_expr;
                        if (loopRank == 1)
                            index_expr = "[0]";
                        else if (depNumber > 2)
                          index_expr = "[id_" + toStr(cuda_letter_first) + " + " + "id_" + toStr(cuda_letter_second) + " * Emin" + "]";
                        else
                          index_expr = "[id_" + toStr(cuda_letter_first) + "]";
                        if (op.second)
                            caseKernelBody += indent + epsGrid + index_expr + " = " + op.first + "(" + epsGrid + index_expr + ", " + red.arrayName + ");\n";
                        else
                            caseKernelBody += indent + epsGrid + index_expr + " = " + epsGrid + index_expr + " " + op.first + " " + red.arrayName + ";\n";
                    }
                }
            }
        }
    }

    const CudaReq &req;
    const KernelDesc &kernelDesc;
    int depNumber;

    bool isAutoTfm;
    bool isPrepareDiag;

    std::string caseKernelFormalParams;
    std::string caseKernelFormalParams_arrays;
    std::string caseKernelFormalParams_scalars;
    std::string caseKernelBody;

    std::string coords;
    std::string globalThreadIdx;

    std::map<std::string, std::vector<std::string> > dvmCoefs;
    std::map<std::string, std::map<std::string, std::string> > dvmDiagInfos;
    std::map<std::string, std::string> redGrid;
};

class AcrossCudaHostCaseHandlerHelper : public HandlerHelper<AcrossCudaHostCaseHandlerHelper> {
public:
    AcrossCudaHostCaseHandlerHelper(const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                                    const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                                    const CudaReq &aReq, int aDepNumber, const std::string &aCaseHandlerName) :
      HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f),
      prohibitedGlobal(prohibitedGlobal), req(aReq), depNumber(aDepNumber), caseHandlerName(aCaseHandlerName) {
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
        int caseHandlerNameNumber = (1 << depNumber) - 1;
    }
    std::pair<std::string, std::string> generate() {
        // TODO: process templates, also for HOST handlers
        std::string handlerTemplateDecl;
        // Generate kernels
        std::vector<KernelDesc> kernelsAvailable;
        kernelsAvailable.push_back(KernelDesc(caseHandlerName, "int"));
        kernelsAvailable.push_back(KernelDesc(caseHandlerName, "long long"));
        int case_number = (1 << depNumber) - 1;
        kernelsAvailable.at(0).kernelName = req.handlerName + "_kernel_" + toStr(case_number) + "_case_int";
        kernelsAvailable.at(0).regsVar = kernelsAvailable.at(0).kernelName + "_regs";
        kernelsAvailable.at(1).kernelName = req.handlerName + "_kernel_" + toStr(case_number) + "_case_llong";
        kernelsAvailable.at(1).regsVar = kernelsAvailable.at(1).kernelName + "_regs";
        caseHandlerBody += indent + "/* "+ toStr(depNumber) + " dependencies */\n";
        genGeneralPrecomputePart(kernelsAvailable);
        if (depNumber == 1)
            generateOneDepCase(kernelsAvailable);
        else if (depNumber == 2)
            generateTwoDepCase(kernelsAvailable);
        else if (depNumber >= 3)
            generateOtherDepCase(kernelsAvailable);
        std::string caseHandlerText;
        for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
            AcrossCudaKernelCaseHandlerHelper h(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f, req, kernelsAvailable[i], depNumber);
            caseHandlerText += h.generate();
        }
        caseHandlerText += (blankCtx.isCPlusPlus() ? handlerTemplateDecl : "extern \"C\" ");
        caseHandlerText += "void " + caseHandlerName + "(" + caseHandlerFormalParams + ") {\n";
        caseHandlerText += caseHandlerBody;
        caseHandlerText += "}\n\n";
        return std::make_pair(caseHandlerText, cudaInfoText);
    }

    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        dependencyMask = getUniqueName("dependencyMask", &prohibitedLocal, &prohibitedGlobal);
        tmpVar = getUniqueName("tmpVar", &prohibitedLocal, &prohibitedGlobal);
        tmpV = getUniqueName("tmpV", &prohibitedLocal, &prohibitedGlobal);
        idxs = getUniqueName("idxs", &prohibitedLocal, &prohibitedGlobal);
        stream = getUniqueName("stream", &prohibitedLocal, &prohibitedGlobal);
        kernelIndexT = getUniqueName("kernelIndexT", &prohibitedLocal, &prohibitedGlobal);
        threads = getUniqueName("threads", &prohibitedLocal, &prohibitedGlobal);
        shared_mem = getUniqueName("shared_mem", &prohibitedLocal, &prohibitedGlobal);
        blocks = getUniqueName("blocks", &prohibitedLocal, &prohibitedGlobal);
        q = getUniqueName("q", &prohibitedLocal, &prohibitedGlobal);
        num_of_red_blocks = getUniqueName("num_of_red_blocks", &prohibitedLocal, &prohibitedGlobal);
        diag = getUniqueName("diag", &prohibitedLocal, &prohibitedGlobal);
        elem = getUniqueName("elem", &prohibitedLocal, &prohibitedGlobal);
        Allmin = getUniqueName("Allmin", &prohibitedLocal, &prohibitedGlobal);
        Emin = getUniqueName("Emin", &prohibitedLocal, &prohibitedGlobal);
        Emax = getUniqueName("Emax", &prohibitedLocal, &prohibitedGlobal);
        var1 = getUniqueName("var1", &prohibitedLocal, &prohibitedGlobal);
        var2 = getUniqueName("var2", &prohibitedLocal, &prohibitedGlobal);
        var3 = getUniqueName("var3", &prohibitedLocal, &prohibitedGlobal);
        SE = getUniqueName("SE", &prohibitedLocal, &prohibitedGlobal);
        for (std::vector<ClauseReduction>::const_iterator i = curPragma.reductions.begin(), ei = curPragma.reductions.end(); i != ei; ++i) {
            redGrid[i->arrayName] = getUniqueName(i->arrayName + "_grid", &prohibitedLocal, &prohibitedGlobal);
            if (i->isLoc())
                redGrid[i->locName] = getUniqueName(i->locName + "_grid", &prohibitedLocal, &prohibitedGlobal);
        }
    }

    void genUniqueNamesForArrayImp(const VarState &varState, const std::set<std::string> &prohibitedGlobal, const std::set<std::string> &prohibitedLocal) {
        HandlerHelper::genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
        dvmDevHeaders[varState.name] = getUniqueName(varState.name + "_devHdr", &prohibitedLocal, &prohibitedGlobal);
        dvmCoefs[varState.name].clear();
        for (int j = 0; j < varState.headerArraySize; j++)
            dvmCoefs[varState.name].push_back(getUniqueName(varState.name + "_hdr" + toStr(j), &prohibitedLocal, &prohibitedGlobal));
    }

    void prepareArrayParameterImp(const VarState &varState, unsigned idx) {
        const std::string &refName = varState.name;
        std::string hdrName = dvmHeaders.find(refName)->second;
        std::string devHdrName = dvmDevHeaders.find(refName)->second;
        caseHandlerFormalParams += ", DvmType " + hdrName + "[]";
        std::string arrType = varState.baseTypeStr;
        if (opts.autoTfm) {
              std::string extendedParamsName = refName + "_extendedParams";
              std::string typeOfTransformName = refName + "_typeOfTransform";
              caseHandlerBody += indent + "dvmh_loop_autotransform_C(" + loop_ref + ", " + hdrName + ");\n";
              caseHandlerBody += indent + arrType + " *" + refName + " = (" + arrType + "*)" +
                      "dvmh_get_natural_base_C(" + device_num + ", " + hdrName + ");\n";
              caseHandlerBody += indent + "DvmType " + devHdrName + "[" + toStr(varState.headerArraySize) + "];\n";
              caseHandlerBody += indent + "DvmType " + extendedParamsName + "[7];\n";
              caseHandlerBody += indent + "DvmType " + typeOfTransformName + " = dvmh_fill_header_C(" +
                      device_num + ", " + refName + ", " + hdrName + ", " + devHdrName + ", " + extendedParamsName + ");\n";
              caseHandlerBody += indent + "assert(" + typeOfTransformName + " == 0 || " +
                      typeOfTransformName + " == 1 || " +
                      typeOfTransformName + " == 2);\n";
              caseHandlerBody += "\n";
              // Create kernel actual parameters
              caseKernelActualArrayParams += ", " + refName;
              if (varState.rank > 1)
                  for (int i = 1; i <= varState.rank; ++i)
                      caseKernelActualArrayParams += ", " + devHdrName + "[" + toStr(i) + "]";
              if (opts.autoTfm && depNumber > 1) {
                  std::string extendedParamsName = refName + "_extendedParams";
                  caseKernelActualArrayParams += ", " + refName + "_typeOfTransform";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[0]";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[3]";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[2]";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[5]";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[1]";
                  caseKernelActualArrayParams += ", " + extendedParamsName + "[4]";
//                caseKernelActualArrayParams += ", " + extendedParamsName + "[6]";
              }
        } else {
            caseHandlerBody += indent + "DvmType " + devHdrName + "[" + toStr(varState.headerArraySize) + "];\n";
            caseHandlerBody += indent + arrType + " *" + refName + " = " +
                    "(" + arrType + "*)dvmh_get_natural_base_C(" + device_num + ", " + hdrName + ");\n";
            caseHandlerBody += indent + tmpVar + " = dvmh_fill_header_C(" + device_num +
                    ", " + refName + ", " + hdrName + ", " + devHdrName + ", 0);\n";
            caseHandlerBody += indent + "assert(" + tmpVar + " == 0 || " + tmpVar + " == 1);\n";
            caseHandlerBody += "\n";
            // Create kernel actual parameters
            caseKernelActualArrayParams += ", " + refName;
            for (int i = 1; i <= varState.rank - 1; ++i) {
                caseKernelActualArrayParams += ", " + devHdrName + "[" + toStr(i) + "]";
            }
        }
    }

    void prepareScalarParameterImp(const VarState &varState, unsigned idx) {
        const std::string &refName = varState.name;
        const std::string &ptrName = scalarPtrs.find(refName)->second;
        caseHandlerFormalParams += ", " + varState.baseTypeStr + "* " + ptrName;
        caseHandlerBody += indent + varState.baseTypeStr + " *" + refName + " = (" +  varState.baseTypeStr + "*)dvmh_get_device_addr_C(" + device_num + ", " + ptrName + ");\n";
        // Create kernel actual parameters
        caseKernelActualScalarParams += ", " + refName;
    }

private:
    void genGeneralPrecomputePart(const std::vector<KernelDesc> &kernelsAvailable) {
        if (!opts.autoTfm) {
            caseHandlerBody += indent + "DvmType " + tmpVar + ";\n";
            caseHandlerBody += "\n";
        }
        caseHandlerBody += indent + "/* Loop references and device number */\n";
        caseHandlerBody += indent + "DvmType " + loop_ref + " = *" + pLoopRef + ";\n";
        caseHandlerBody += indent + "DvmType " + device_num + " = dvmh_loop_get_device_num_C(" + loop_ref + ");\n";
        caseHandlerBody += "\n";

        caseHandlerFormalParams += "DvmType* " + pLoopRef;
        caseHandlerBody += indent + "/* Parameters */\n";
        prepareParameters();
        caseHandlerFormalParams += ", DvmType " + dependencyMask;
        caseHandlerBody += "\n";

        caseHandlerBody += indent + "/* Supplementary variables for loop handling */\n";
        caseHandlerBody += indent + "DvmType " + boundsLow + "[" + toStr(loopRank) + "]";
        caseHandlerBody += ", " + boundsHigh + "[" + toStr(loopRank) + "]";
        caseHandlerBody += ", " + loopSteps + "[" + toStr(loopRank) + "];\n";
        caseHandlerBody += indent + "DvmType " + idxs + "[" + toStr(loopRank) + "];\n";
        caseHandlerBody += indent + "cudaStream_t " + stream + ";\n";
        caseHandlerBody += "\n";

        // Prepare general kernel parameters
        for (int i = 0; i < loopRank; ++i) {
            caseKernelActualLoopStepParams += ", " + loopSteps + "[" + toStr(i) + "]";
            caseKernelActualIdxParams += ", " + idxs + "[" + toStr(i) + "]";
        }

        caseHandlerBody += indent + "/* Choose index type for CUDA kernel */\n";
        caseHandlerBody += indent + "int " + kernelIndexT + " = dvmh_loop_guess_index_type_C(" + loop_ref + ");\n";
        caseHandlerBody += indent + "if (" + kernelIndexT + " == rt_LONG) " + kernelIndexT + " = (sizeof(long) <= sizeof(int) ? rt_INT : rt_LLONG);\n";
        caseHandlerBody += indent + "assert(" + kernelIndexT + " == rt_INT || " + kernelIndexT + " == rt_LLONG);\n";
        caseHandlerBody += "\n";
        caseHandlerBody += indent + "/* Fill loop bounds */\n";
        caseHandlerBody += indent + "dvmh_loop_fill_bounds_C(" + loop_ref + ", " + boundsLow + ", " + boundsHigh + ", " + loopSteps + ");\n";
        caseHandlerBody += indent + "dvmh_change_filled_bounds2_C(" + boundsLow + ", " + boundsHigh + ", " + loopSteps + ", " + toStr(loopRank) + ", " +
                dependencyMask + ", " + idxs + ");\n";
        caseHandlerBody += "\n";

        caseHandlerBody += indent + "/* Get CUDA configuration parameters */\n";
        int threads_conf_x = 0;
        int threads_conf_y = 0;
        int threads_conf_z = 0;
        getDefaultCudaBlock(threads_conf_x, threads_conf_y, threads_conf_z, depNumber, loopRank - depNumber, opts.autoTfm);
        std::string threadsConf = toStr(threads_conf_x) + ", " + toStr(threads_conf_y) + ", " + toStr(threads_conf_z);
        caseHandlerBody += indent + "dim3 " + threads + " = dim3(" + threadsConf + ");\n";
        if (loopRank > depNumber) {
            caseHandlerBody += "#ifdef CUDA_FERMI_ARCH\n";
            caseHandlerBody += indent + "DvmType " + shared_mem + " = 8;\n";
            caseHandlerBody += "#else\n";
        }
        caseHandlerBody += indent + "DvmType " + shared_mem + " = 0;\n";
        if (loopRank > depNumber)
            caseHandlerBody += "#endif\n";
        for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
            std::string regsVar = kernelsAvailable[i].regsVar;
            std::string rtIndexT = kernelsAvailable[i].rtIndexT;
            cudaInfoText += "#ifdef " + toUpper(regsVar) + "\n";
            cudaInfoText += indent + "DvmType " + regsVar + " = " + toUpper(regsVar) + ";\n";
            cudaInfoText += "#else\n";
            cudaInfoText += indent + "DvmType " + regsVar + " = 0;\n";
            cudaInfoText += "#endif\n";
            caseHandlerBody += indent + "extern DvmType " + regsVar + ";\n";
            caseHandlerBody += indent + "if (" + kernelIndexT + " == " + rtIndexT + ") dvmh_loop_cuda_get_config_C(" + loop_ref + ", " + shared_mem + ", " +
                    regsVar + ", &" +
                    threads + ", &" + stream + ", &" + shared_mem + ");\n";
        }
        caseHandlerBody += "\n";
    }

    void prepareReduction() {
        for (int i = 0; i < (int)curPragma.reductions.size(); ++i) {
            const ClauseReduction &red = curPragma.reductions[i];
            std::string epsGrid = redGrid[red.arrayName];
            llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
            VarDecl *vd = vdi == localDecls.end() ? 0 : vdi->second;
            checkIntErrN(vd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
            VarState varState;
            fillVarState(vd, blankCtx.isCPlusPlus(), comp, &varState);
            // TODO: a diagnostic must point of the original pragma with reduction
            checkUserErrN(!varState.isArray, fileName.c_str(), curPragma.line, 4417, varState.name.c_str());
            caseHandlerBody += indent + varState.baseTypeStr + " " + red.arrayName + ";\n";
            caseHandlerBody += indent + varState.baseTypeStr + " *" + epsGrid + ";\n";
            caseKernelActualReductionParams += ", " + red.arrayName + ", " + epsGrid;
            if (red.isLoc()) {
                std::string locGrid = redGrid[red.locName];
                llvm::StringMap<VarDecl *>::const_iterator vdi = localDecls.find(red.arrayName);
                VarDecl *lvd = vdi == localDecls.end() ? 0 : vdi->second;
                checkIntErrN(lvd, 97, red.arrayName.c_str(), fileName.c_str(), curPragma.line);
                VarState locVarState;
                fillVarState(lvd, blankCtx.isCPlusPlus(), comp, &locVarState);
                // TODO: a diagnostic must point of the original pragma with reduction
                checkUserErrN(!locVarState.isArray, fileName.c_str(), curPragma.line, 4417, locVarState.name.c_str());
                caseHandlerBody += indent + locVarState.baseTypeStr + " " + red.locName + ";\n";
                caseHandlerBody += indent + locVarState.baseTypeStr + " *" + locGrid + ";\n";
                caseHandlerBody += indent + "dvmh_loop_cuda_register_red_C(" + loop_ref + ", " + toStr(i + 1) + ", (void **)&" + epsGrid + ", (void **)&" +
                        locGrid + ");\n";
                caseHandlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                        ", " + (locVarState.isArray ? "" : "&") + red.locName + ");\n";
            } else {
                caseHandlerBody += indent + "dvmh_loop_cuda_register_red_C(" + loop_ref + ", " + toStr(i + 1) + ", (void **)&" + epsGrid + ", 0);\n";
                caseHandlerBody += indent + "dvmh_loop_red_init_C(" + loop_ref + ", " + toStr(i + 1) + ", " + (varState.isArray ? "" : "&") + red.arrayName +
                        ", 0);\n";
            }
            caseHandlerBody += indent + "dvmh_loop_cuda_red_prepare_C(" + loop_ref + ", " + toStr(i + 1) + ", " + num_of_red_blocks + ", 1);\n";
        }
    }

    void endReduction() {
        if (curPragma.reductions.size() > 0) {
            caseHandlerBody += indent + "/* Finish reduction */\n";
            for (int i = 0; i < (int)curPragma.reductions.size(); ++i) {
                caseHandlerBody += indent + "dvmh_loop_cuda_red_finish_C(" + loop_ref + ", " + toStr(i + 1) + ");\n";
            }
        }
    }

    void genKernelCalls(const std::vector<KernelDesc> &kernelsAvailable, const std::string &caseKernelActualParams) {
        for (int i = 0; i < (int)kernelsAvailable.size(); i++) {
            std::string caseKernelName = kernelsAvailable[i].kernelName;
            std::string rtIndexT = kernelsAvailable[i].rtIndexT;
            caseHandlerBody += indent + "if (" + kernelIndexT + " == " + rtIndexT + ") " +
                    caseKernelName + "<<<" + blocks + ", " + threads + ", " + shared_mem + ", " + stream + ">>>(" +
                    caseKernelActualParams + ");\n";
        }
    }

    void generateOneDepCase(const std::vector<KernelDesc> &kernelsAvailable) {
        assert(depNumber == 1 && "Current method is for across for one dependent dimension only!");
        int nCudaDims = std::min(loopRank - depNumber, 3);
        caseHandlerBody += indent + "/* Calculate computation distribution parameters */\n";
        // Allocating CUDA threads
        for (int i = 0; i < nCudaDims; ++i) {
            char letter = 'x' + i;
            caseHandlerBody += indent + "DvmType num_" + letter + " = threads." + toStr(letter) + ";\n";
        }
        caseHandlerBody += "\n";
        caseHandlerBody += indent + "dim3 " + blocks + " = dim3(1, 1, 1);\n";
        std::string caseKernelActualBaseParams_base;
        std::string caseKernelActualNumElemIndepParams;
        for (int i = 0; i < loopRank; ++i) {
            caseHandlerBody += indent + "int " + "base_" + toStr(i) + " = " + boundsLow + "[" + toStr(i) + "];\n";
            caseKernelActualBaseParams_base += ", base_" + toStr(i);
            if (i >= 1) {
                char letter = 'x' + std::min(i - 1, 2);
                std::string letter_suffix = toStr(letter) + (i > 2 ? "_" + toStr(i - 3) : "");
                std::string boundsLow_i = boundsLow + "[" + toStr(i) + "]";
                std::string boundsHigh_i = boundsHigh + "[" + toStr(i) + "]";
                std::string loopSteps_i = loopSteps + "[" + toStr(i) + "]";
                caseHandlerBody += indent + "int " + "num_elem_" + letter_suffix + " = " +
                        "(abs(" + boundsLow_i + " - " + boundsHigh_i + ")" + " + " +
                        "abs(" + loopSteps_i + "))" + " / " + "abs(" + loopSteps_i + ")" +
                        ";\n";
                caseKernelActualNumElemIndepParams += ", num_elem_" + letter_suffix;
            }
            if (i == 1 || i == 2) {
                char letter = 'x' + std::min(i - 1, 2);
                caseHandlerBody += indent + blocks + "." + toStr(letter) + " = " + "(" +
                        "num_elem_" + toStr(letter) + " + " + "num_" + toStr(letter) + " - 1" + ")" +
                        " / " + "num_" + toStr(letter) +
                        ";\n";
                caseHandlerBody += indent + threads + "." + toStr(letter) + " = " + "num_" + toStr(letter) + ";\n";
            }
        }
        if (loopRank >= 4) {
            char letter = 'z';
            caseHandlerBody += indent + "int " + "num_elem_" + toStr(letter) + " = ";
            int zsize = loopRank - 3;
            for (int i = 0; i < zsize; ++i) {
                if (i != 0)
                    caseHandlerBody += " * ";
                caseHandlerBody += "num_elem_" + toStr(letter) + "_" + toStr(i);
            }
            caseHandlerBody += ";\n";
            caseHandlerBody += indent + blocks + "." + toStr(letter) + " = " + "(" +
                    "num_elem_" + toStr(letter) + " + " + "num_" + toStr(letter) + " - 1" + ")" +
                    " / " + "num_" + toStr(letter) +
                    ";\n";
            caseHandlerBody += indent + threads + "." + toStr(letter) + " = " + "num_" + toStr(letter) + ";\n";
        }
        {
            std::string boundsLow_0 = boundsLow + "[0]";
            std::string boundsHigh_0 = boundsHigh + "[0]";
            std::string loopSteps_0 = loopSteps + "[0]";
            caseHandlerBody += indent + boundsHigh_0 + " = " +
                    "(abs(" + boundsLow_0 + " - " + boundsHigh_0 + ")" + " + " +
                    "abs(" + loopSteps_0 + "))" + " / " + "abs(" + loopSteps_0 + ")" +
                    ";\n";
        }
        caseHandlerBody += "\n";
        // Start reduction
        if (curPragma.reductions.size() > 0) {
            caseHandlerBody += indent + "/* Reductions-related stuff */\n";
            caseHandlerBody += indent + "DvmType " + num_of_red_blocks + " = " + "blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z";
            if (depNumber < loopRank)
              caseHandlerBody += " / 32";
            caseHandlerBody +=  ";\n";
            prepareReduction();
            caseHandlerBody += "\n";
        }
        caseHandlerBody += indent + "for (int " + tmpV + " = 0; " + tmpV + " < " + boundsHigh + "[0]; " +
                "base_0 = base_0 + " + loopSteps + "[0], " + tmpV + " = " + tmpV + " + 1) {\n";
        indent += blankCtx.getIndentStep();
        // Generate kernel call.
       std::string caseKernelActualParams;
        caseKernelActualParams += caseKernelActualArrayParams;
        caseKernelActualParams += caseKernelActualScalarParams;
        caseKernelActualParams += caseKernelActualReductionParams;
        caseKernelActualParams += caseKernelActualNumElemIndepParams;
        caseKernelActualParams += caseKernelActualBaseParams_base;
        caseKernelActualParams += caseKernelActualLoopStepParams;
        caseKernelActualParams += caseKernelActualIdxParams;
        trimList(caseKernelActualParams);
        genKernelCalls(kernelsAvailable, caseKernelActualParams);
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        endReduction();
    }

    void generateTwoDepCase(const std::vector<KernelDesc> &kernelsAvailable) {
        assert(depNumber == 2 && "Current method is for across for two dependent dimension only!");
        int nCudaDims = 1 + (loopRank >= depNumber + 1 ? 1 : 0) + (loopRank > depNumber + 1 ? 1 : 0);
        caseHandlerBody += indent + "/* Calculate computation distribution parameters */\n";
        // Allocating CUDA threads.
        for (int i = 0; i < nCudaDims; ++i) {
            char letter = 'x' + i;
            caseHandlerBody += indent + "DvmType num_" + letter + " = threads." + toStr(letter) + ";\n";
        }
        caseHandlerBody += "\n";
        // Dependent dims.
        for (int i = 0; i < depNumber; ++i)
            caseHandlerBody += indent + "int M" + toStr(i) + " = (" + boundsHigh + "[" + toStr(i) + "] - " +
                    boundsLow + "[" + toStr(i) + "]) / " + loopSteps + "[" + toStr(i) + "] + 1;\n";
        // Independent dims.
        std::string caseKernelActualNumeElemIndepParams;
        for (int i = depNumber; i < loopRank; ++i) {
            caseHandlerBody += indent + "DvmType num_elem_" + toStr(i) + " = (" + boundsHigh + "[" + toStr(i) + "] - " +
                    boundsLow + "[" + toStr(i) + "]) / " + loopSteps + "[" + toStr(i) + "] + 1;\n";
            caseKernelActualNumeElemIndepParams += ", num_elem_" + toStr(i);
        }
        if (nCudaDims >= 2)
            caseHandlerBody += indent + "DvmType num_elem_y = num_elem_2;\n";
        if (nCudaDims == 3) {
            caseHandlerBody += indent + "DvmType num_elem_z = ";
            for (int i = 3; i < loopRank; ++i) {
                if (i != 3)
                    caseHandlerBody += ", ";
                caseHandlerBody += "num_elem_" + toStr(i);
            }
            caseHandlerBody += ";\n";
        }
        caseHandlerBody += "\n";
        // Determine blocks.
        caseHandlerBody += indent + "dim3 " + blocks + " = dim3(";
        for (int i = 0; i < nCudaDims; ++i) {
            char letter = 'x' + i;
            if (i != 0)
                caseHandlerBody += ", ";
            caseHandlerBody += "num_" + toStr(letter);
        }
        caseHandlerBody += ");\n";
        for (int i = 1; i < nCudaDims; ++i) {
            char letter = 'x' + i;
            caseHandlerBody += indent + "blocks." + toStr(letter) + " = (num_elem_" + toStr(letter) +
                    " + num_" + toStr(letter) + " - 1) / num_" + toStr(letter) + ";\n";
        }
        caseHandlerBody += "\n";
        caseHandlerBody += indent + "DvmType " + q + " = min(M0, M1);\n";
        caseHandlerBody += "\n";
        // Start reduction.
        if (curPragma.reductions.size() > 0) {
            caseHandlerBody += indent + "/* Reductions-related stuff */\n";
            if (loopRank == depNumber) {
                caseHandlerBody += indent + "DvmType " + num_of_red_blocks + " = " + q + ";\n";
            }  else {
                caseHandlerBody += indent + "DvmType " + num_of_red_blocks + " = " + "((" + q + " + num_x - 1)/ num_x) * blocks.y * blocks.z * num_x * threads.y * threads.z / 32;\n";
            }
            prepareReduction();
            caseHandlerBody += "\n";
        }
        std::string caseKernelActualBaseParams;
        for (int i = 0; i < loopRank; ++i) {
            caseHandlerBody += indent + "int " + "base_" + toStr(i) + " = " + boundsLow + "[" + toStr(i) + "];\n";
            caseKernelActualBaseParams += ", base_" + toStr(i);
        }
        caseHandlerBody += "\n";
        // GPU execution.
        caseHandlerBody += indent + "/* GPU execution */\n";
        // First part, first loop.
        caseHandlerBody += indent + "int " + diag + " = 1;\n";
        caseHandlerBody += indent + "while (" + diag + " <= " + q + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + diag + " + num_x - 1) / num_x;\n";
        // Generate kernel call.
        {
            std::string caseKernelActualParams;
            caseKernelActualParams += caseKernelActualArrayParams;
            caseKernelActualParams += caseKernelActualScalarParams;
            caseKernelActualParams += caseKernelActualReductionParams;
            caseKernelActualParams += ", " + diag;
            caseKernelActualParams += caseKernelActualNumeElemIndepParams;
            caseKernelActualParams += caseKernelActualBaseParams;
            caseKernelActualParams += caseKernelActualLoopStepParams;
            caseKernelActualParams += caseKernelActualIdxParams;
            trimList(caseKernelActualParams);
            genKernelCalls(kernelsAvailable, caseKernelActualParams);
        }
        caseHandlerBody += indent + "base_0 = base_0 + " + loopSteps + "[0];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        // Second part.
        caseHandlerBody += indent + "int " + elem + ";\n";
        caseHandlerBody += indent + "if (M0 < M1) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_0 = base_0 - " + loopSteps + "[0];\n";
        caseHandlerBody += indent + "base_1 = " + boundsLow + "[1] + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = 0;\n";
        // Second part, first loop.
        caseHandlerBody += indent + "while (" + diag + " < M1 - M0) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + q + " + num_x - 1) / num_x;\n";
        {
            std::string caseKernelActualParams;
            caseKernelActualParams += caseKernelActualArrayParams;
            caseKernelActualParams += caseKernelActualScalarParams;
            caseKernelActualParams += caseKernelActualReductionParams;
            caseKernelActualParams += ", " + q;
            caseKernelActualParams += caseKernelActualNumeElemIndepParams;
            caseKernelActualParams += caseKernelActualBaseParams;
            caseKernelActualParams += caseKernelActualLoopStepParams;
            caseKernelActualParams += caseKernelActualIdxParams;
            trimList(caseKernelActualParams);
            genKernelCalls(kernelsAvailable, caseKernelActualParams);
        }
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += indent + diag + " = " + q + " + (M1 - M0) + 1;\n";
        caseHandlerBody += indent + elem + " = " + q + " - 1;\n";
        // Second part, second loop.
        caseHandlerBody += indent + "while (" + diag + " < M0 + M1) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + elem + " + num_x - 1) / num_x;\n";
        {
            std::string caseKernelActualParams;
            caseKernelActualParams += caseKernelActualArrayParams;
            caseKernelActualParams += caseKernelActualScalarParams;
            caseKernelActualParams += caseKernelActualReductionParams;
            caseKernelActualParams += ", " + elem;
            caseKernelActualParams += caseKernelActualNumeElemIndepParams;
            caseKernelActualParams += caseKernelActualBaseParams;
            caseKernelActualParams += caseKernelActualLoopStepParams;
            caseKernelActualParams += caseKernelActualIdxParams;
            trimList(caseKernelActualParams);
            genKernelCalls(kernelsAvailable, caseKernelActualParams);
        }
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        caseHandlerBody += indent + elem + " = " + elem + " - 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        // Third part, first loop.
        caseHandlerBody += indent + diag + " = 0;\n";
        caseHandlerBody += indent + "while (" + diag + " < M0 - M1) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + q + " + num_x - 1) / num_x;\n";
        {
            std::string caseKernelActualParams;
            caseKernelActualParams += caseKernelActualArrayParams;
            caseKernelActualParams += caseKernelActualScalarParams;
            caseKernelActualParams += caseKernelActualReductionParams;
            caseKernelActualParams += ", " + q;
            caseKernelActualParams += caseKernelActualNumeElemIndepParams;
            caseKernelActualParams += caseKernelActualBaseParams;
            caseKernelActualParams += caseKernelActualLoopStepParams;
            caseKernelActualParams += caseKernelActualIdxParams;
            trimList(caseKernelActualParams);
            genKernelCalls(kernelsAvailable, caseKernelActualParams);
        }
        caseHandlerBody += indent + "base_0 = base_0 + " + loopSteps + "[0];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += indent + "base_0 = base_0 - " + loopSteps + "[0];\n";
        caseHandlerBody += indent + "base_1 = " + boundsLow + "[1] + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = " + q + " + (M0 - M1) + 1;\n";
        caseHandlerBody += indent + elem + " = " + q + " - 1;\n";
        // Third part, second loop.
        caseHandlerBody += indent + "while (" + diag + " < M0 + M1) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + elem + " + num_x - 1) / num_x;\n";
        {
            std::string caseKernelActualParams;
            caseKernelActualParams += caseKernelActualArrayParams;
            caseKernelActualParams += caseKernelActualScalarParams;
            caseKernelActualParams += caseKernelActualReductionParams;
            caseKernelActualParams += ", " + elem;
            caseKernelActualParams += caseKernelActualNumeElemIndepParams;
            caseKernelActualParams += caseKernelActualBaseParams;
            caseKernelActualParams += caseKernelActualLoopStepParams;
            caseKernelActualParams += caseKernelActualIdxParams;
            trimList(caseKernelActualParams);
            genKernelCalls(kernelsAvailable, caseKernelActualParams);
        }
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        caseHandlerBody += indent + elem + " = " + elem + " - 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        endReduction();
    }

    void generateOtherDepCase(const std::vector<KernelDesc> &kernelsAvailable) {
        assert(depNumber >= 3 && "Current method is for across for more than three dependent dimensions only!");
        int n_cuda_dims = 2 + (loopRank > depNumber ? 1 : 0);
        caseHandlerBody += indent + "/* Calculate computation distribution parameters */\n";
        // Allocating CUDA threads
        for (int i = 0; i < n_cuda_dims; ++i) {
            char letter = 'x' + i;
            caseHandlerBody += indent + "DvmType num_" + letter + " = threads." + toStr(letter) + ";\n";
        }
        caseHandlerBody += "\n";
        // Dependent dims
        for (int i = 0; i < 3; ++i)
            caseHandlerBody += indent + "int M" + toStr(i) + " = (" + boundsHigh + "[" + toStr(i) + "] - " +
                    boundsLow + "[" + toStr(i) + "]) / " + loopSteps + "[" + toStr(i) + "] + 1;\n";
        caseHandlerBody += indent + "int " + Allmin + " = min(min(M0, M1), M2);\n";
        caseHandlerBody += indent + "int " + Emin + " = min(M0, M1);\n";
        caseHandlerBody += indent + "int " + Emax + " = min(M0, M1) + abs(M0 - M1) + 1;\n";
        // Independent dims
        std::string caseKernelActualNumElemIndepParams;
        for (int i = depNumber; i < loopRank; ++i) {
            caseHandlerBody += indent + "DvmType num_elem_" + toStr(i) + " = (" + boundsHigh + "[" + toStr(i) + "] - " +
                    boundsLow + "[" + toStr(i) + "]) / " + loopSteps + "[" + toStr(i) + "] + 1;\n";
            caseKernelActualNumElemIndepParams += ", num_elem_" + toStr(i);
        }
        if (n_cuda_dims == 3) {
            caseHandlerBody += indent + "DvmType num_elem_z = ";
            for (int i = 3; i < loopRank; ++i) {
                if (i != 3)
                    caseHandlerBody += " * ";
                caseHandlerBody += "num_elem_" + toStr(i);
            }
            caseHandlerBody += + ";\n";
        }
        caseHandlerBody += "\n";
        caseHandlerBody += indent + "int " + var1 + " = 1;\n";
        caseHandlerBody += indent + "int " + var2 + " = 0;\n";
        caseHandlerBody += indent + "int " + var3 + " = 0;\n";
        caseHandlerBody += indent + "int " + diag + " = 1;\n";
        caseHandlerBody += indent + "int " + SE + " = 1;\n";
        // Determine blocks
        caseHandlerBody += indent + "dim3 " + blocks + " = dim3(";
        for (int i = 0; i < n_cuda_dims; ++i) {
            char letter = 'x' + i;
            if (i != 0)
                caseHandlerBody += ", ";
            caseHandlerBody += "num_" + toStr(letter);
        }
        caseHandlerBody += ");\n";
        caseHandlerBody += "\n";
        if (n_cuda_dims == 3) {
            caseHandlerBody += indent + "blocks.z = (num_elem_z + num_z - 1) / num_z;\n";
            caseHandlerBody += "\n";
        }
        // Start reduction
        if (curPragma.reductions.size() > 0) {
            caseHandlerBody += indent + "/* Reductions-related stuff */\n";
            if (loopRank == depNumber)
                caseHandlerBody += indent + "DvmType " + num_of_red_blocks + " = " + Emin + " * max(max(M0, M1), M2);\n";
            else
            caseHandlerBody += indent + "DvmType " + num_of_red_blocks + " = ((" + Emin + " + num_x - 1) / num_x)" +
                               " * ((max(max(M0, M1), M2) + num_y - 1) / num_y) * blocks.z * num_x * num_y * num_z / 32;\n";
            prepareReduction();
            caseHandlerBody += "\n";
        }
        // GPU execution
        caseHandlerBody += indent + "/* GPU execution */\n";
        if (depNumber > 3) {
            for (int i = 3; i < depNumber; ++i)
                caseHandlerBody += indent + "int " + "base_" + toStr(i) + " = " + boundsLow + "[" + toStr(i) + "];\n";
            caseHandlerBody += indent + "while (base_3 <= " + boundsHigh + "[3]) {\n";
            indent += blankCtx.getIndentStep();
            caseHandlerBody += indent + var1 + " = 1;\n";
            caseHandlerBody += indent + var2 + " = 0;\n";
            caseHandlerBody += indent + var3 + " = 0;\n";
            caseHandlerBody += indent + diag + " = 1;\n";
            caseHandlerBody += indent + SE + " = 1;\n";
        }
        for (int i = 0; i <= 2; ++i)
            caseHandlerBody += indent + "int " + "base_" + toStr(i) + " = " + boundsLow + "[" + toStr(i) + "];\n";
        for (int i = depNumber; i < loopRank; ++i)
            caseHandlerBody += indent + "int " + "base_" + toStr(i) + " = " + boundsLow + "[" + toStr(i) + "];\n";
        // Generate kernel actual parameters.
        std::string caseKernelActualBaseParams;
        for (int i = 0; i < loopRank; ++i)
            caseKernelActualBaseParams += ", base_" + toStr(i);
        std::string caseKernelActualParams;
        caseKernelActualParams += caseKernelActualArrayParams;
        caseKernelActualParams += caseKernelActualScalarParams;
        caseKernelActualParams += caseKernelActualReductionParams;
        caseKernelActualParams += caseKernelActualBaseParams;
        caseKernelActualParams += caseKernelActualLoopStepParams;
        caseKernelActualParams += ", " + diag + ", " + SE;
        caseKernelActualParams += ", " + var1 + ", " + var2 + ", " + var3;
        caseKernelActualParams += ", " + Emax + ", " + Emin;
        caseKernelActualParams += ", min(M0, M1)";
        caseKernelActualParams += ", M0 > M1";
        caseKernelActualParams += caseKernelActualNumElemIndepParams;
        caseKernelActualParams += caseKernelActualIdxParams;
        trimList(caseKernelActualParams);
        caseHandlerBody += "\n";
        // First part, first loop
        caseHandlerBody += indent + diag + " = 1;\n";
        caseHandlerBody += indent + "while (" + diag + " <= " + Allmin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + diag + " + num_x - 1) / num_x;\n";
        caseHandlerBody += indent + "blocks.y = (" + diag + " + num_y - 1) / num_y;\n";
        genKernelCalls(kernelsAvailable, caseKernelActualParams);
        caseHandlerBody += indent + "base_2 = base_2 + " + loopSteps + "[2];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        // Second stage
        caseHandlerBody += indent + var1 + " = 0;\n";
        caseHandlerBody += indent + var2 + " = 0;\n";
        caseHandlerBody += indent + var3 + " = 1;\n";
        caseHandlerBody += indent + "if (M2 > " + Emin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_0 = " + boundsLow + "[0] * (M0 <= M1) + " + boundsLow + "[1] * (M0 > M1);\n";
        caseHandlerBody += indent + "base_1 = " + boundsLow + "[1] * (M0 <= M1) + " + boundsLow + "[0] * (M0 > M1);\n";
        caseHandlerBody += indent + diag + " = Allmin + 1;\n";
        caseHandlerBody += indent + "while (" + diag + " - 1 != M2) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + Emin + " + num_x - 1) / num_x;\n";
        caseHandlerBody += indent + "blocks.y = (" + diag + " + num_y - 1) / num_y;\n";
        genKernelCalls(kernelsAvailable, caseKernelActualParams);
        caseHandlerBody += indent + "base_2 = base_2 + " + loopSteps + "[2];\n";
        caseHandlerBody += indent + diag + " = " + diag + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        // Third stage.
        caseHandlerBody += indent + diag + " = M2;\n";
        caseHandlerBody += indent + "blocks.y = (" + diag + " + num_y - 1) / num_y;\n";
        caseHandlerBody += indent + "blocks.x = (" + Emin + " + num_x - 1) / num_x;\n";
        caseHandlerBody += indent + SE + " = 2;\n";
        caseHandlerBody += indent + "base_0 = (" + boundsLow + "[0] + " + loopSteps + "[0]) * (M0 <= M1) + (" +
                boundsLow + "[1] + " + loopSteps + "[1]) * (M0 > M1);\n";
        caseHandlerBody += indent + "base_1 = " + boundsLow + "[1] * (M0 <= M1) + " +
                boundsLow + "[0] * (M0 > M1);\n";
        caseHandlerBody += indent + "base_2 = " + boundsLow + "[2] + " + loopSteps + "[2] * (M2 - 1);\n";
        caseHandlerBody += indent + "while (M0 + M1 - " + Allmin + " != " + SE + " - 1) {\n";
        indent += blankCtx.getIndentStep();
        genKernelCalls(kernelsAvailable, caseKernelActualParams);
        caseHandlerBody += indent + "base_0 = base_0 + " + loopSteps + "[0] * (M0 <= M1) + " + loopSteps + "[1] * (M0 > M1);\n";
        caseHandlerBody += indent + SE + " = " + SE + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        // Forth stage.
        caseHandlerBody += indent + var1 + " = 0;\n";
        caseHandlerBody += indent + var2 + " = 1;\n";
        caseHandlerBody += indent + var3 + " = 0;\n";
        caseHandlerBody += indent + diag + " = " + Allmin + " - 1;\n";
        caseHandlerBody += indent + "base_0 = " + boundsLow + "[0] + " + loopSteps + "[0] * (M0 - 1);\n";
        caseHandlerBody += indent + "base_1 = " + boundsLow + "[1] * (M0 > M1) + base_1 * (M0 <= M1);\n";
        caseHandlerBody += indent + "if (M0 > M1 && M2 <= " + Emin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1] + abs(" + Emin + " - M2);\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "if (M0 <= M1 && M2 <= " + Emin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "if (" + loopSteps + "[1] > 0) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1] + " + Emax + " - " + Emin + " - 1 + " + "abs(" + Emin + " - M2);\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1] - " + Emax + " + " + Emin + " + 1 + " + "M2 - " + Emin + ";\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "if (M0 > M1 && M2 > " + Emin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1];\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "if (M0 <= M1 && M2 > " + Emin + ") {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "if (" + loopSteps + "[1] > 0) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1] + " + Emax + " - " + Emin + " - 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "} else {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1] - " + Emax + " + " + Emin + " + 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += indent + "while (" + diag + " != 0) {\n";
        indent += blankCtx.getIndentStep();
        caseHandlerBody += indent + "blocks.x = (" + diag + " + num_x - 1) / num_x;\n";
        caseHandlerBody += indent + "blocks.y = (" + diag + " + num_y - 1) / num_y;\n";
        genKernelCalls(kernelsAvailable, caseKernelActualParams);
        caseHandlerBody += indent + SE + " = " + SE + " + 1;\n";
        caseHandlerBody += indent + "base_1 = base_1 + " + loopSteps + "[1];\n";
        caseHandlerBody += indent + diag + " = " + diag + " - 1;\n";
        indent = subtractIndent(indent);
        caseHandlerBody += indent + "}\n";
        caseHandlerBody += "\n";
        if (depNumber > 3) {
            caseHandlerBody += indent + "base_3 = base_3 + " + loopSteps + "[3];\n";
            indent = subtractIndent(indent);
            caseHandlerBody += indent + "}\n";
            for (int i = 4; i < depNumber; ++i)
                caseHandlerBody += indent + "base_" + toStr(i) + " = base_" + toStr(i) + " + " + loopSteps + "[" + toStr(i) + "];\n";
        }
        endReduction();
    }

private:
    const std::set<std::string> &prohibitedGlobal;
    const CudaReq &req;
    int depNumber;

    std::string caseHandlerName;
    std::string caseHandlerFormalParams;
    std::string caseHandlerBody;
    std::string cudaInfoText;

    std::string caseKernelActualArrayParams;
    std::string caseKernelActualScalarParams;
    std::string caseKernelActualLoopStepParams;
    std::string caseKernelActualIdxParams;
    std::string caseKernelActualNumeElemIndepParams;
    std::string caseKernelActualReductionParams;

    std::string dependencyMask;
    std::string tmpVar;
    std::string tmpV;
    std::string idxs;
    std::string stream;
    std::string kernelIndexT;
    std::string threads;
    std::string shared_mem;
    std::string blocks;
    std::string q;
    std::string num_of_red_blocks;
    std::string diag;
    std::string elem;
    std::string Allmin;
    std::string Emin;
    std::string Emax;
    std::string var1;
    std::string var2;
    std::string var3;
    std::string SE;
    std::map<std::string, std::string> dvmDevHeaders;
    std::map<std::string, std::vector<std::string> > dvmCoefs;
    std::map<std::string, std::string> redGrid;
};

class AcrossCudaHostHandlerHelper : public HandlerHelper<AcrossCudaHostHandlerHelper> {
public:
    AcrossCudaHostHandlerHelper(const ConverterOptions &opts, CompilerInstance &comp, Rewriter &rwr, const HandlerFileContext &blankCtx,
                      const std::set<std::string> &prohibitedGlobal, const PragmaHandlerStub &curPragma, const FunctionDecl &f,
                      const CudaReq &aReq) :
      HandlerHelper(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f), req(aReq), prohibitedGlobal(prohibitedGlobal), handlerFormalParams(f.getNumParams() + 1 - curPragma.weirdRmas.size()) {
        std::set<std::string> prohibitedLocal = extractProhibitedNames(comp, curPragma, f.getBody());
        genUniqueNames(prohibitedGlobal, prohibitedLocal);
    }

    std::pair<std::string, std::string> generate() {
        // TODO: process templates, also for HOST handlers
        std::string handlerTemplateSpec, handlerTemplateDecl;
        assert(0 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[0] = "DvmType *" + pLoopRef;
        typicalCaseHandlerAcutalParams += pLoopRef;
        prepareParameters();
        typicalCaseHandlerAcutalParams += ", " + dependency_mask;
        handlerBody += indent + "/* Get number of dependencies */\n";
        handlerBody += indent + "int " + dependency_mask + " = dvmh_loop_get_dependency_mask_C(*" + pLoopRef + ")" + ";\n";
        handlerBody += indent + "int " + dependency_mask_tmp + " = " + dependency_mask + ";\n";
        handlerBody += indent + "int " + dependency_num + " = 0;\n";
        handlerBody += indent + "while(" + dependency_mask_tmp + ") {\n";
        indent += blankCtx.getIndentStep();
        handlerBody += indent + dependency_mask_tmp + " &= (" + dependency_mask_tmp + " - 1);\n";
        handlerBody += indent + "++" + dependency_num + ";\n";
        indent = subtractIndent(indent);
        handlerBody += indent + "}\n\n";
        handlerBody += indent + "/* Run the corresponding handler */\n";
        std::pair<std::string, std::string> caseHandlers;
        for (int i = curPragma.minAcross; i <= curPragma.maxAcross; ++i) {
            if (i != curPragma.minAcross)
                handlerBody += indent + "else if (" + dependency_num + " == " + toStr(i) + ") {\n";
            else
                handlerBody += indent + "if (" + dependency_num + " == " + toStr(i) + ") {\n";
            indent += blankCtx.getIndentStep();
            // Insert case handler call.
            int caseHandlerNameNumber = (1 << i) - 1;
            std::string caseHandlerName = req.handlerName + "_" + toStr(caseHandlerNameNumber) + "_case";
            handlerBody += indent + caseHandlerName + "(" + typicalCaseHandlerAcutalParams + ");\n";
            indent = subtractIndent(indent);
            handlerBody += indent + "}\n";
            // Generate case handler.
            AcrossCudaHostCaseHandlerHelper h(opts, comp, rwr, blankCtx, prohibitedGlobal, curPragma, f, req, i, caseHandlerName);
            std::pair<std::string, std::string> res = h.generate();
            caseHandlers.first += res.first;
            caseHandlers.second += res.second;
        }
        std::string handlerText = caseHandlers.first + (blankCtx.isCPlusPlus() ? handlerTemplateDecl : "extern \"C\" ") + "void " +
                                  req.handlerName + "(" + llvm::join(handlerFormalParams.begin(), handlerFormalParams.end(), ", ") + ") {\n" + handlerBody + "}\n";
        return std::make_pair(handlerText, caseHandlers.second);
    }

    void genUniqueNamesForInternalImp(const std::set<std::string>& prohibitedGlobal, const std::set<std::string>& prohibitedLocal) {
        HandlerHelper::genUniqueNamesForInternalImp(prohibitedGlobal, prohibitedLocal);
        dependency_mask = getUniqueName("dependency_mask", &prohibitedGlobal, &prohibitedLocal);
        dependency_mask_tmp = getUniqueName("dependency_mask_tmp", &prohibitedGlobal, &prohibitedLocal);
        dependency_num = getUniqueName("dependency_num", &prohibitedGlobal, &prohibitedLocal);
    }

    void genUniqueNamesForArrayImp(const VarState &varState, const std::set<std::string> &prohibitedGlobal, const std::set<std::string> &prohibitedLocal) {
        HandlerHelper::genUniqueNamesForArrayImp(varState, prohibitedGlobal, prohibitedLocal);
        dvmDevHeaders[varState.name] = getUniqueName(varState.name + "_devHdr", &prohibitedLocal, &prohibitedGlobal);
        dvmCoefs[varState.name].clear();
        for (int j = 0; j < varState.headerArraySize; j++)
            dvmCoefs[varState.name].push_back(getUniqueName(varState.name + "_hdr" + toStr(j), &prohibitedLocal, &prohibitedGlobal));
    }

    void prepareArrayParameterImp(const VarState &varState, unsigned idx) {
        const std::string &refName = varState.name;
        std::string hdrName = dvmHeaders.find(refName)->second;
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1] = "DvmType " + hdrName + "[]";
        typicalCaseHandlerAcutalParams += ", " + hdrName;
    }

    void prepareScalarParameterImp(const VarState &varState, unsigned idx) {
        const std::string &refName = varState.name;
        const std::string &ptrName = scalarPtrs.find(refName)->second;
        assert(idx + 1 < handlerFormalParams.size() && "Too many formal parmeters in a handler!");
        handlerFormalParams[idx + 1] = varState.baseTypeStr + " *" + ptrName;
        typicalCaseHandlerAcutalParams += ", " + ptrName;
    }

private:
    const CudaReq &req;
    const std::set<std::string> &prohibitedGlobal;

    std::string dependency_mask;
    std::string dependency_mask_tmp;
    std::string dependency_num;
    std::map<std::string, std::string> dvmDevHeaders;
    std::map<std::string, std::vector<std::string> > dvmCoefs;

    std::vector<std::string> handlerFormalParams;
    std::string typicalCaseHandlerAcutalParams;
    std::string handlerBody;
};
}

// Blank2CudaVisitor

bool Blank2CudaVisitor::VisitFunctionDecl(FunctionDecl *f) {
    // Handler cannnot be in macro.
    if (f->getLocStart().isMacroID())
      return true;
    FileID fileID = srcMgr.getFileID(f->getLocStart());
    SourceLocation incLoc = srcMgr.getIncludeLoc(fileID);
    if (incLoc.isValid())
      return true;
    int pragmaLine = srcMgr.getExpansionLineNumber(f->getLocStart()) - 1;
    const PragmaHandlerStub *curPragma = ph->getPragmaAtLine(pragmaLine);
    bool isHandler = curPragma != 0;
    // TODO: remove includes and DVMH_VARIABLE_ARRAY_SIZE
    // TODO: mark functions to execute on device or remove them if there are no calls
    if (!withHeading && firstHandler && isHandler) {
        firstHandler = false;
        if (curPragma->line > 2)
            rewr.RemoveText(SourceRange(srcMgr.getLocForStartOfFile(fileID), srcMgr.translateLineCol(fileID, curPragma->line, 1).getLocWithOffset(-1)));
    }
    std::map<std::string, CudaReq>::const_iterator reqItr = blankCtx.getCudaReqMap().find(f->getName().str());
    if (reqItr != blankCtx.getCudaReqMap().end()) {
        checkIntErrN(isHandler, 915, f->getName().data());
        removePragma(*curPragma, fileID, rewr);
        // TODO: process templates in C++ sources
        std::pair<std::string, std::string> handler;
        if (curPragma->minAcross != 0 || curPragma->maxAcross != 0) {
            AcrossCudaHostHandlerHelper h(opts, comp, rewr, blankCtx, prohibitedNames, *curPragma, *f, reqItr->second);
            handler = h.generate();
        } else {
            CudaHostHandlerHelper h(opts, comp, rewr, blankCtx, prohibitedNames, *curPragma, *f, reqItr->second);
            handler = h.generate();
        }
        rewr.ReplaceText(f->getSourceRange(), handler.first);
        cudaInfoText += handler.second;
    } else if (isHandler) {
        SourceRange sr(srcMgr.translateLineCol(fileID, curPragma->line, 1), f->getLocEnd());
        rewr.RemoveText(sr);
    }
    return true;
}

// BlankHandlerConverter

BlankHandlerConverter::BlankHandlerConverter(const SourceFileContext &fileCtx):
    opts(fileCtx.getProjectCtx().getOptions()), blankCtx(fileCtx.getBlankContext()), prohibitedNames(dvm0CHelper.getAllPossibleNames()) {
    prohibitedNames.insert(fileCtx.seenMacroNames.begin(), fileCtx.seenMacroNames.end());
}

std::string BlankHandlerConverter::genRmas(const std::string &src) {
    std::string res = src;
    std::string fn = prepareFile(src);
    {
        PassContext passCtx(fn);
        Rewriter *rewr = passCtx.getRewr();
        BlankPragmaHandler *pragmaHandler = new BlankPragmaHandler(*passCtx.getCompiler());
        passCtx.getPP()->AddPragmaHandler(pragmaHandler);
        BlankRemoteConsumer astConsumer(*passCtx.getCompiler(), *rewr, pragmaHandler, prohibitedNames);
        passCtx.parse(&astConsumer);
        const RewriteBuffer *rewriteBuf = rewr->getRewriteBufferFor(rewr->getSourceMgr().getMainFileID());
        if (rewriteBuf)
            res = std::string(rewriteBuf->begin(), rewriteBuf->end());
    }
    remove(fn.c_str());
    return res;
}

std::string BlankHandlerConverter::genHostHandlers(const std::string &src, bool withHeading) {
    std::string res = src;
    std::string fn = prepareFile(src);
    {
        PassContext passCtx(fn);
        Rewriter *rewr = passCtx.getRewr();
        BlankPragmaHandler *pragmaHandler = new BlankPragmaHandler(*passCtx.getCompiler());
        passCtx.getPP()->AddPragmaHandler(pragmaHandler);
        Blank2HostConsumer astConsumer(opts, *passCtx.getCompiler(), blankCtx, *rewr, pragmaHandler, prohibitedNames, dvm0CHelper, withHeading);
        passCtx.parse(&astConsumer);
        const RewriteBuffer *rewriteBuf = rewr->getRewriteBufferFor(rewr->getSourceMgr().getMainFileID());
        if (rewriteBuf) {
            res.clear();
            std::string macros = dvm0CHelper.genDvm0cUndefText() + dvm0CHelper.genDvm0cDefText();
            if (!macros.empty()) {
                if (opts.extraComments)
                    res += "/* Supplementary macros */\n";
                res += macros;
            }
            res += std::string(rewriteBuf->begin(), rewriteBuf->end());
            if (!macros.empty())
                res += dvm0CHelper.genDvm0cUndefText();
        }
    }
    remove(fn.c_str());
    return res;
}

std::pair<std::string, std::string> BlankHandlerConverter::genCudaHandlers(const std::string &src, bool withHeading) {
    std::string handlers = src;
    std::string info;
    std::string fn = prepareFile(src);
    {
        PassContext passCtx(fn);
        Rewriter *rewr = passCtx.getRewr();
        BlankPragmaHandler *pragmaHandler = new BlankPragmaHandler(*passCtx.getCompiler());
        passCtx.getPP()->AddPragmaHandler(pragmaHandler);
        Blank2CudaConsumer astConsumer(opts, *passCtx.getCompiler(), blankCtx, *rewr, pragmaHandler, prohibitedNames, withHeading);
        passCtx.parse(&astConsumer);
        const RewriteBuffer *rewriteBuf = rewr->getRewriteBufferFor(rewr->getSourceMgr().getMainFileID());
        if (rewriteBuf) {
            handlers = std::string(rewriteBuf->begin(), rewriteBuf->end());
            info = astConsumer.getCudaInfo().str();
        }
    }
    remove(fn.c_str());
    return std::make_pair(handlers, info);
}

std::string BlankHandlerConverter::prepareFile(const std::string &src) {
    const char fnTempl[] = "CDVMH_XXXXXX";
    char *fn1 = myStrDup(fnTempl);
#ifndef WIN32
    char *tmpRes = mktemp(fn1);
#else
    char *tmpRes = _mktemp(fn1);
#endif
    checkIntErrN(tmpRes && tmpRes[0], 916);
    std::string fn(fn1);
    delete[] fn1;
    fn += (blankCtx.isCPlusPlus() ? ".cpp" : ".c");
    FILE *f = fopen(fn.c_str(), "wb");
    checkIntErrN(f, 916);
    fwrite(src.c_str(), src.size(), 1, f);
    fclose(f);
    return fn;
}
}
