#pragma once

#include "aux_visitors.h"

namespace cdvmh {

struct MyVarState: public VarState {
    bool isWeirdRma;
    MyVarState(): isWeirdRma(false) {}
};

struct ClauseBlankRma {
    std::string origName;
    std::string substName;
    std::vector<std::string> indexExprs;
    unsigned nonConstRank;
    std::vector<int> appearances;
};

struct PragmaHandlerStub {
    int line;
    std::set<std::string> dvmArrays;
    std::set<std::string> regArrays;
    std::set<std::string> scalars;
    std::vector<LoopVarDesc> loopVars;
    std::vector<ClauseReduction> reductions;
    std::set<std::string> privates;
    std::vector<std::string> weirdRmas;
    std::vector<ClauseBlankRma> rmas;
    int minAcross;
    int maxAcross;
};

class BlankPragmaHandler: public PragmaHandler {
public:
    explicit BlankPragmaHandler(CompilerInstance &aComp): PragmaHandler("dvm"), comp(aComp) {}
public:
#if CLANG_VERSION_MAJOR > 8
    void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer, Token &FirstToken) CDVMH_OVERRIDE;
#else
    void HandlePragma(Preprocessor& PP, PragmaIntroducerKind Introducer, Token& FirstToken) CDVMH_OVERRIDE;
#endif
public:
    PragmaHandlerStub *getPragmaAtLine(int line) {
      std::map<int, PragmaHandlerStub *>::iterator i = pragmas.find(line);
        if (i != pragmas.end())
            return i->second;
        else
            return 0;
    }

    const PragmaHandlerStub *getPragmaAtLine(int line) const {
      std::map<int, PragmaHandlerStub *>::const_iterator i = pragmas.find(line);
        if (i != pragmas.end())
            return i->second;
        else
            return 0;
    }
protected:
    CompilerInstance &comp;
    std::map<int, PragmaHandlerStub *> pragmas;
};

class BlankRemoteVisitor: public RecursiveASTVisitor<BlankRemoteVisitor> {
    typedef RecursiveASTVisitor<BlankRemoteVisitor> base;
public:
    explicit BlankRemoteVisitor(CompilerInstance &aComp, Rewriter &aRewr, BlankPragmaHandler *aPh, const std::set<std::string> &seenMacroNames):
            comp(aComp), rewr(aRewr), srcMgr(rewr.getSourceMgr()), ph(aPh), seenMacroNames(seenMacroNames), curPragma(0), parLoopBodyStmt(0),
            inParLoopBody(false), parLoopBodyExprCounter(0)
    {}
public:
    bool VisitFunctionDecl(FunctionDecl *f);
    bool TraverseStmt(Stmt *s);
    bool VisitArraySubscriptExpr(ArraySubscriptExpr *e);
protected:
    CompilerInstance &comp;
    Rewriter &rewr;
    SourceManager &srcMgr;

    BlankPragmaHandler *ph;
    const std::set<std::string> &seenMacroNames;

    PragmaHandlerStub *curPragma;
    Stmt *parLoopBodyStmt;
    bool inParLoopBody;
    int parLoopBodyExprCounter;

    struct RmaInfo {
      std::vector<ClauseBlankRma>::const_iterator itr;
      int curSubscriptIdx;
      llvm::SmallVector<llvm::SmallString<8>, 4> replacement;
    } curRma;
};

class BlankRemoteConsumer: public ASTConsumer {
public:
    explicit BlankRemoteConsumer(CompilerInstance &comp, Rewriter &rewr, BlankPragmaHandler *ph, const std::set<std::string> &seenMacroNames):
            rv(comp, rewr, ph, seenMacroNames) {}
public:
    void HandleTranslationUnit(ASTContext &Ctx) CDVMH_OVERRIDE {
        rv.TraverseDecl(Ctx.getTranslationUnitDecl());
    }
protected:
    BlankRemoteVisitor rv;
};

class Blank2HostVisitor: public RecursiveASTVisitor<Blank2HostVisitor> {
    typedef RecursiveASTVisitor<Blank2HostVisitor> base;
public:
    Blank2HostVisitor(const ConverterOptions &aOpts, CompilerInstance &aComp, const HandlerFileContext &aBlankCtx, Rewriter &aRewr, const BlankPragmaHandler *aPh,
                      const std::set<std::string> &aProhibitedNames, Dvm0CHelper &aDvm0CHelper, bool aWithHeading):
            opts(aOpts), comp(aComp), blankCtx(aBlankCtx), rewr(aRewr), ph(aPh), prohibitedNames(aProhibitedNames), dvm0CHelper(aDvm0CHelper), withHeading(aWithHeading),
            srcMgr(rewr.getSourceMgr()), firstHandler(true), isOpenMPIncluded(false) {}
public:
    bool VisitFunctionDecl(FunctionDecl *f);
    bool TraverseFunctionDecl(FunctionDecl *f);
protected:
    const ConverterOptions &opts;
    CompilerInstance &comp;
    Rewriter &rewr;
    const BlankPragmaHandler *ph;
    const HandlerFileContext &blankCtx;
    const std::set<std::string> &prohibitedNames;
    Dvm0CHelper &dvm0CHelper;
    bool withHeading;

    SourceManager &srcMgr;
    bool firstHandler;
    bool isOpenMPIncluded;
};

class Blank2HostConsumer: public ASTConsumer {
public:
    Blank2HostConsumer(const ConverterOptions &opts, CompilerInstance &comp, const HandlerFileContext &blankCtx, Rewriter &rewr, const BlankPragmaHandler *ph,
                       const std::set<std::string> &prohibitedNames, Dvm0CHelper &dvm0CHelper, bool withHeading):
            rv(opts, comp, blankCtx, rewr, ph, prohibitedNames, dvm0CHelper, withHeading) {}
public:
    void HandleTranslationUnit(ASTContext &Ctx) CDVMH_OVERRIDE {
        rv.TraverseDecl(Ctx.getTranslationUnitDecl());
    }
protected:
    Blank2HostVisitor rv;
};

class Blank2CudaVisitor: public RecursiveASTVisitor<Blank2CudaVisitor> {
    typedef RecursiveASTVisitor<Blank2CudaVisitor> base;
public:
    Blank2CudaVisitor(const ConverterOptions &aOpts, CompilerInstance &aComp, const HandlerFileContext &aBlankCtx, Rewriter &aRewr, const BlankPragmaHandler *aPh,
                      const std::set<std::string> &aProhibitedNames, bool aWithHeading):
            opts(aOpts), comp(aComp), blankCtx(aBlankCtx), rewr(aRewr), ph(aPh), prohibitedNames(aProhibitedNames), withHeading(aWithHeading),
            srcMgr(rewr.getSourceMgr()), firstHandler(true) {}
public:
    bool VisitFunctionDecl(FunctionDecl *f);

    llvm::StringRef getCudaInfo() const { return cudaInfoText; }
protected:
    const ConverterOptions &opts;
    CompilerInstance &comp;
    Rewriter &rewr;
    const BlankPragmaHandler *ph;
    const HandlerFileContext &blankCtx;
    const std::set<std::string> &prohibitedNames;
    bool withHeading;

    SourceManager &srcMgr;
    bool firstHandler;

    std::string cudaInfoText;
};

class Blank2CudaConsumer: public ASTConsumer {
public:
    Blank2CudaConsumer(const ConverterOptions &opts, CompilerInstance &comp, const HandlerFileContext &blankCtx, Rewriter &rewr, const BlankPragmaHandler *ph,
                       const std::set<std::string> &prohibitedNames, bool withHeading):
            rv(opts, comp, blankCtx, rewr, ph, prohibitedNames, withHeading) {}
public:
    void HandleTranslationUnit(ASTContext &Ctx) CDVMH_OVERRIDE {
        rv.TraverseDecl(Ctx.getTranslationUnitDecl());
    }

    llvm::StringRef getCudaInfo() const { return rv.getCudaInfo(); }

  protected:
    Blank2CudaVisitor rv;
};

class BlankHandlerConverter {
public:
    explicit BlankHandlerConverter(const SourceFileContext &aFileCtx);
public:
    std::string genRmas(const std::string &src);
    std::string genHostHandlers(const std::string &src, bool withHeading = true);
    std::pair<std::string, std::string> genCudaHandlers(const std::string &src, bool withHeading = true);
protected:
    std::string prepareFile(const std::string &src);
protected:
    const HandlerFileContext &blankCtx;
    const ConverterOptions &opts;
    Dvm0CHelper dvm0CHelper;
    std::set<std::string> prohibitedNames;
};
}
