#pragma once

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Pragma.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Basic/Version.h>

#include "pragmas.h"
#include "file_ctx.h"

using clang::PragmaHandler;
using clang::CompilerInstance;
using clang::Rewriter;
using clang::Preprocessor;
using clang::PragmaIntroducerKind;
using clang::Token;
using clang::FileID;
#if CLANG_VERSION_MAJOR > 8
using clang::PragmaIntroducer;
#endif

namespace cdvmh {

#if CLANG_VERSION_MAJOR > 8
class ExternalPreprocessor;
using PreprocessorImpl = ExternalPreprocessor;
#else
typedef Preprocessor PreprocessorImpl;
#endif

class DvmPragmaHandler: public PragmaHandler {
public:
    explicit DvmPragmaHandler(SourceFileContext &aFileCtx, CompilerInstance &aComp, Rewriter &aRewr): PragmaHandler("dvm"), fileCtx(aFileCtx),
            projectCtx(fileCtx.getProjectCtx()), comp(aComp), rewr(aRewr) {}
public:
#if CLANG_VERSION_MAJOR > 8
    void HandlePragma(Preprocessor& PP, PragmaIntroducer Introducer, Token& FirstToken) CDVMH_OVERRIDE;
#else
    void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer, Token &FirstToken) CDVMH_OVERRIDE;
#endif
protected:
    DerivedAxisRule parseDerivedAxisRule(PreprocessorImpl &PP, Token &Tok);
    DistribRule parseDistribRule(PreprocessorImpl &PP, Token &Tok);
    AlignAxisRule parseAlignAxisRule(PreprocessorImpl &PP, Token &Tok, std::map<std::string, int> nameToAxis, bool parLoopFlag);
    AlignRule parseAlignRule(PreprocessorImpl &PP, Token &Tok, bool parLoopFlag = false);
    std::vector<SlicedArray> parseSubarrays(PreprocessorImpl &PP, Token &Tok);
    MyExpr readExpr(PreprocessorImpl &PP, Token &Tok, std::string stopTokens = "");
    std::pair<MyExpr, MyExpr> readRange(PreprocessorImpl &PP, Token &Tok);
    ClauseRemoteAccess parseOneRma(PreprocessorImpl &PP, Token &Tok, const std::map<std::string, int> &nameToAxis);
protected:
    SourceFileContext &fileCtx;
    ProjectContext &projectCtx;
    CompilerInstance &comp;
    Rewriter &rewr;
    DvmPragma *curPragma;
    FileID fileID;
};

}
