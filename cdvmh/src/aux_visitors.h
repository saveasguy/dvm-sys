#pragma once

#include <map>
#include <vector>
#include <string>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Rewrite/Core/Rewriter.h>
#if CLANG_VERSION_MAJOR > 10
#include <clang/Basic/FileManager.h>
#endif
#include "pragmas.h"
#include "file_ctx.h"
#include "utils.h"

using namespace clang; // Sad, but there are too many used from this namespace

namespace cdvmh {

class CollectNamesVisitor: public RecursiveASTVisitor<CollectNamesVisitor> {
    typedef RecursiveASTVisitor<CollectNamesVisitor> base;
public:
    explicit CollectNamesVisitor(CompilerInstance &aComp): comp(aComp), ignoreDepth(0) {}
public:
    bool VisitVarDecl(VarDecl *vd);
    bool VisitTypedefDecl(TypedefDecl *td);
    bool VisitFunctionDecl(FunctionDecl *f);
    bool TraverseFunctionDecl(FunctionDecl *f);
    bool VisitRecordDecl(RecordDecl *d);
    bool TraverseRecordDecl(RecordDecl *d);
    bool VisitDeclRefExpr(DeclRefExpr *e);
public:
    std::set<std::string> getNames() { return names; }
protected:
    CompilerInstance &comp;

    int ignoreDepth;
    std::set<std::string> names;
};

class MacroCollector: public PPCallbacks {
public:
    explicit MacroCollector(SourceFileContext &aFileCtx): fileCtx(aFileCtx) {}
public:
    void MacroDefined(const Token &MacroNameTok, const MacroDirective *MD) CDVMH_OVERRIDE {
        fileCtx.seenMacroNames.insert(MacroNameTok.getIdentifierInfo()->getName().str());
    }
protected:
    SourceFileContext &fileCtx;
};

class PPDirectiveCollector: public PPCallbacks {
public:
    explicit PPDirectiveCollector(CompilerInstance &comp): comp(comp) {}
public:
#if CLANG_VERSION_MAJOR > 15
    void InclusionDirective(SourceLocation HashLoc,
                            const Token &IncludeTok,
                            StringRef FileName,
                            bool IsAngled,
                            CharSourceRange FilenameRange,
                            OptionalFileEntryRef File,
                            StringRef SearchPath,
                            StringRef RelativePath,
                            const Module *Imported,
                            SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE {
#elif CLANG_VERSION_MAJOR > 14
    void InclusionDirective(SourceLocation HashLoc,
                            const Token &IncludeTok,
                            StringRef FileName,
                            bool IsAngled,
                            CharSourceRange FilenameRange,
                            Optional<FileEntryRef> File,
                            StringRef SearchPath,
                            StringRef RelativePath,
                            const Module *Imported,
                            SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE {
#elif CLANG_VERSION_MAJOR > 6
    void InclusionDirective(SourceLocation HashLoc,
                            const Token &IncludeTok,
                            StringRef FileName,
                            bool IsAngled,
                            CharSourceRange FilenameRange,
                            const FileEntry *File,
                            StringRef SearchPath,
                            StringRef RelativePath,
                            const Module *Imported,
                            SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE {
#else
    void InclusionDirective(SourceLocation HashLoc,
                            const Token &IncludeTok,
                            StringRef FileName,
                            bool IsAngled,
                            CharSourceRange FilenameRange,
                            const FileEntry *File,
                            StringRef SearchPath,
                            StringRef RelativePath,
                            const Module *Imported) CDVMH_OVERRIDE {
#endif
        addDirective(HashLoc);
    }
    void PragmaDirective(SourceLocation Loc,
                               PragmaIntroducerKind Introducer) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
#if (CLANG_VERSION_MAJOR > 3 || CLANG_VERSION_MINOR > 6)
    void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
                       StringRef Str) CDVMH_OVERRIDE {
#else
    void PragmaComment(SourceLocation Loc, const IdentifierInfo *Kind,
                       const std::string &Str) CDVMH_OVERRIDE {
#endif
      addDirective(Loc);
    }
#if (CLANG_VERSION_MAJOR > 3 || CLANG_VERSION_MINOR > 6)
    void PragmaDetectMismatch(SourceLocation Loc, StringRef Name,
                              StringRef Value) CDVMH_OVERRIDE {
#else
    void PragmaDetectMismatch(SourceLocation Loc, const std::string &Name,
                              const std::string &Value) CDVMH_OVERRIDE {
#endif

      addDirective(Loc);
    }
    void PragmaDebug(SourceLocation Loc, StringRef DebugType) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
    void PragmaMessage(SourceLocation Loc, StringRef Namespace,
                       PragmaMessageKind Kind, StringRef Str) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
    void PragmaDiagnosticPush(SourceLocation Loc,
                              StringRef Namespace) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
    void PragmaDiagnosticPop(SourceLocation Loc,
                             StringRef Namespace) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
    // TODO: Does not compile on 3.4.2, add proper variant when needed
    //virtual void PragmaDiagnostic(SourceLocation Loc, StringRef Namespace,
    //                              diag::Severity mapping, StringRef Str) {
    //    addDirective(Loc);
    //}
#if CLANG_VERSION_MAJOR > 13
    void PragmaWarning(SourceLocation Loc, PragmaWarningSpecifier WarningSpec,
                       ArrayRef<int> Ids) CDVMH_OVERRIDE {
#else
    void PragmaWarning(SourceLocation Loc, StringRef WarningSpec,
                       ArrayRef<int> Ids) CDVMH_OVERRIDE {
#endif
        addDirective(Loc);
    }
    void MacroDefined(const Token &MacroNameTok,
                              const MacroDirective *MD) CDVMH_OVERRIDE {
        addDirective(MacroNameTok.getLocation());
    }
#if CLANG_VERSION_MAJOR > 4
    void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                        const MacroDirective *Undef) CDVMH_OVERRIDE {
#elif CLANG_VERSION_MINOR > 6
    void MacroUndefined(const Token &MacroNameTok,
                        const MacroDefinition &MD) CDVMH_OVERRIDE {
#else
    void MacroUndefined(const Token &MacroNameTok,
                        const MacroDirective *MD) CDVMH_OVERRIDE {
#endif
        addDirective(MacroNameTok.getLocation());
    }
    // TODO: Does not compile on 3.4.2, add proper variant when needed
    //virtual void If(SourceLocation Loc, SourceRange ConditionRange,
    //                ConditionValueKind ConditionValue) {
    //    addDirective(Loc);
    //}
    //virtual void Elif(SourceLocation Loc, SourceRange ConditionRange,
    //                  ConditionValueKind ConditionValue, SourceLocation IfLoc) {
    //    addDirective(Loc);
    //}
#if CLANG_VERSION_MAJOR > 4 || CLANG_VERSION_MINOR > 6
    void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) CDVMH_OVERRIDE {
#else
    void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDirective *MD) CDVMH_OVERRIDE {
#endif
        addDirective(Loc);
    }
#if CLANG_VERSION_MAJOR > 4 || CLANG_VERSION_MINOR > 6
   void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) CDVMH_OVERRIDE {
#else
    void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDirective *MD) CDVMH_OVERRIDE {
#endif
        addDirective(Loc);
    }
    void Else(SourceLocation Loc, SourceLocation IfLoc) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
    void Endif(SourceLocation Loc, SourceLocation IfLoc) CDVMH_OVERRIDE {
        addDirective(Loc);
    }
protected:
    void addDirective(SourceLocation beginLoc);
protected:
    CompilerInstance &comp;

    std::map<unsigned, std::map<int, SourceRange> > directives;
};

struct IncludedFile {
    int inclusionCount;
    bool isSystem;
};

struct Inclusion {
    bool valid;
    SourceLocation hashLoc;
    std::pair<int, int> where;
    std::string what;
    FileID whatFID;
    bool isAngled;
    bool isImport;
    bool isIncludeNext;
    bool isSkipped;
    std::string spellingFN;
    std::vector<Inclusion> nested;
    Inclusion(): valid(false) {}
};

class IncludeCollector: public PPCallbacks {
public:
    std::map<std::string, IncludedFile> includedFiles;
    std::vector<Inclusion> inclusions;
public:
    explicit IncludeCollector(SourceFileContext &fileCtx, Preprocessor &PP): fileCtx(fileCtx), PP(PP), ignoreLevel(-1) {}
public:
    void FileChanged(SourceLocation Loc, FileChangeReason Reason, SrcMgr::CharacteristicKind FileType, FileID PrevFID) CDVMH_OVERRIDE;
#if CLANG_VERSION_MAJOR < 10
    void FileSkipped(const FileEntry &ParentFile, const Token &FilenameTok, SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE;
#else
    void FileSkipped(const FileEntryRef& ParentFile, const Token& FilenameTok, SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE;
#endif

#if CLANG_VERSION_MAJOR > 15
    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        OptionalFileEntryRef File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE;
#elif CLANG_VERSION_MAJOR > 14
    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        Optional<FileEntryRef> File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE;
#elif CLANG_VERSION_MAJOR > 6
    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        const FileEntry *File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) CDVMH_OVERRIDE;
#else
    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        const FileEntry *File, StringRef SearchPath, StringRef RelativePath, const Module *Imported) CDVMH_OVERRIDE;
#endif
protected:
    void commitInclusion(SrcMgr::CharacteristicKind FileType);
protected:
    SourceFileContext &fileCtx;
    Preprocessor &PP;

    std::vector<int> activeInclusion;
    Inclusion pendingInclusion;
    int ignoreLevel;
    std::set<FileID> seenFIDs;
};

struct PragmaExpander: public PragmaHandler {
public:
    explicit PragmaExpander(llvm::raw_string_ostream &OS, const char *name = "dvm"): PragmaHandler(name), OS(OS), name(name) {}
#if CLANG_VERSION_MAJOR > 8
    void HandlePragma(Preprocessor& PP, PragmaIntroducer Introducer, Token& PragmaTok) CDVMH_OVERRIDE;
#else
    void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer, Token &PragmaTok) CDVMH_OVERRIDE;
#endif
protected:
    llvm::raw_string_ostream &OS;
    std::string name;
};

class IncludeExpanderAndRewriter {
public:
    explicit IncludeExpanderAndRewriter(SourceFileContext &fileCtx, const std::vector<Inclusion> &inclusions, Rewriter &rewr): fileCtx(fileCtx),
            projectCtx(fileCtx.getProjectCtx()), inclusions(inclusions), rewr(rewr), srcMgr(rewr.getSourceMgr()) {
        mainOutFN = fileCtx.getInputFile().canonicalConverted;
        mainFID = srcMgr.getMainFileID();
    }
public:
    const RewriteBuffer *work() {
        bool isChanged, isHard;
        processInclusions(mainFID, inclusions, isChanged, isHard);
        return rewr.getRewriteBufferFor(mainFID);
    }
protected:
    void processInclusions(FileID parentFID, const std::vector<Inclusion> &incs, bool &isChanged, bool &isHard);
    void processInclusion(const Inclusion &inc, bool &isChanged, bool &isHard);
protected:
    SourceFileContext &fileCtx;
    ProjectContext &projectCtx;
    const std::vector<Inclusion> &inclusions;
    Rewriter &rewr;
    SourceManager &srcMgr;

    FileID mainFID;
    std::string mainOutFN;
};

}

#if CLANG_VERSION_MAJOR > 7
#define getLocEnd getEndLoc
#define getLocStart getBeginLoc
#define getStartLoc getBeginLoc
#endif
