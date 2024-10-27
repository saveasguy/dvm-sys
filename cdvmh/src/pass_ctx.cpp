#include "pass_ctx.h"

#include <cstdio>

#if CLANG_VERSION_MAJOR > 17
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

#include <llvm/ADT/ArrayRef.h>

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Compilation.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Basic/Version.h>

#if CLANG_VERSION_MAJOR > 9
#include <clang/Basic/Builtins.h>
#endif

using clang::DiagnosticOptions;
using clang::TextDiagnosticPrinter;
using clang::DiagnosticIDs;
using clang::DiagnosticsEngine;
using clang::driver::Driver;
using clang::driver::Compilation;
using clang::driver::Command;
using clang::CompilerInvocation;
using clang::FileEntry;
using clang::TargetInfo;
using clang::TargetOptions;
#if CLANG_VERSION_MAJOR > 17
using clang::FileEntryRef;
#endif


#ifdef WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace cdvmh {

void PassContext::clear() {
    if (compiler) {
        for (std::map<std::string, Rewriter *>::iterator it = rewriters.begin(); it != rewriters.end(); it++) {
            delete it->second;
        }
        delete compiler;
        compiler = 0;
    }
}

void PassContext::setup() {
    std::vector<std::string> argvCompile;
    if (getenv("CLANG") && *getenv("CLANG")) {
        argvCompile.push_back(getenv("CLANG"));
    } else {
        std::string strPath("clang");
        if (!isWin) {
            FILE *f = popen("which clang", "r");
            if (f) {
                char path[1024] = {0};
                if (fgets(path, sizeof(path), f)) {
                    strPath = path;
                    if (!strPath.empty() && *strPath.rbegin() == '\n')
                        strPath.resize(strPath.size() - 1);
                }
                pclose(f);
            } else {
                strPath = "/usr/bin/clang";
            }
        }
        argvCompile.push_back(strPath);
    }
    argvCompile.push_back("-x");
    argvCompile.push_back(CPlusPlus ? "c++" : "c");
    if (isWin) {
        argvCompile.push_back("-fms-compatibility");
        argvCompile.push_back("-fms-extensions");
    }
    for (int i = 0; i < (int)includeDirs.size(); i++)
        argvCompile.push_back("-I" + includeDirs[i]);
    if (cdvFile && getenv("dvmdir")) {
        std::string dvmSysIncludeDir = getenv("dvmdir");
        dvmSysIncludeDir += "/include";
        argvCompile.push_back("-I" + dvmSysIncludeDir);
        argvCompile.push_back("-include");
        argvCompile.push_back(dvmSysIncludeDir + "/dvmh_runtime_api.h");
    }
    for (int i = 0; i < (int)addDefines.size(); i++) {
        if (addDefines[i].second.empty())
            argvCompile.push_back("-D" + addDefines[i].first);
        else
            argvCompile.push_back("-D" + addDefines[i].first + "=" + addDefines[i].second);
    }
    if (cdvFile)
        argvCompile.push_back("-D_DVMH");
    for (int i = 0; i < (int)removeDefines.size(); i++)
        argvCompile.push_back("-U" + removeDefines[i]);
    argvCompile.push_back("-E");
    argvCompile.push_back(cdvFileName);
    std::vector<const char *> rawArgs;
    for (int i = 0; i < (int)argvCompile.size(); i++)
        rawArgs.push_back(argvCompile[i].c_str());

    DiagnosticOptions *diagOpts = new DiagnosticOptions;
    TextDiagnosticPrinter *DiagnosticPrinter = new TextDiagnosticPrinter(llvm::errs(), diagOpts);
    DiagnosticsEngine *Diagnostics = new DiagnosticsEngine(new DiagnosticIDs(), diagOpts, DiagnosticPrinter, true); // Owns printer and ids
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 5
    Driver clangDriver(argvCompile[0], llvm::sys::getDefaultTargetTriple(), "a.out", *Diagnostics); // acquires ownership of Diagnostics
#else
    Driver clangDriver(argvCompile[0], llvm::sys::getDefaultTargetTriple(), *Diagnostics); // acquires ownership of Diagnostics
#endif
    Compilation *comp = clangDriver.BuildCompilation(rawArgs);
    if (logLevel >= DEBUG) {
        llvm::errs() << "Clang driver command:";
        comp->getJobs().Print(llvm::errs(), "\n", false);
    }
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 6
    Command *cmd = llvm::dyn_cast<Command>(*comp->getJobs().begin());
#else
    Command *cmd = llvm::dyn_cast<Command>(&*comp->getJobs().begin());
#endif
    assert(cmd);
    const llvm::opt::ArgStringList &cc1Args = cmd->getArguments();

    // XXX: Workaround for non-existent directories problem
    int argsSize = cc1Args.size();
    char **modifiedArgs = new char *[argsSize];
    for (int i = 0; i < argsSize; i++) {
        std::string modified;
        if (strstr(cc1Args[i], "/../"))
            modified = simplifyFileName(cc1Args[i], true);
        else
            modified = cc1Args[i];
        modifiedArgs[i] = myStrDup(modified.c_str());
        if (strcmp(cc1Args[i], modifiedArgs[i]))
            cdvmh_log(DEBUG, "Substituted '%s' with '%s'", cc1Args[i], modifiedArgs[i]);
    }

    // XXX: Workaround for bad llvm installations
    if (!isWin) {
        std::vector<std::string> addPaths;
        FILE *f;
        if (CPlusPlus)
            f = popen("g++ -x c++ -E -v - </dev/null 2>& 1", "r");
        else
            f = popen("gcc -x c -E -v - </dev/null 2>& 1", "r");
        if (f) {
            char lin[2048];
            int state = 0;
            while (fgets(lin, sizeof(lin), f)) {
                if (state == 0) {
                    if (strstr(lin, "#include <...> search starts here:"))
                        state++;
                } else if (state == 1) {
                    if (strstr(lin, "End of search list.")) {
                        state++;
                    } else {
                        int toSkip = 0;
                        while (lin[toSkip] && isspace(lin[toSkip]))
                            toSkip++;
                        int siz = strlen(lin + toSkip);
                        while (siz > 0 && isspace(lin[toSkip + siz - 1]))
                            siz--;
                        lin[toSkip + siz] = 0;
                        std::string newPath = simplifyFileName(lin + toSkip, true);
                        bool toAdd = true;
                        for (int i = 0; i < argsSize; i++)
                            toAdd = toAdd && !(!strcmp(modifiedArgs[i], newPath.c_str()) || getCanonicalFileName(modifiedArgs[i]) ==
                                    getCanonicalFileName(newPath));
                        if (toAdd)
                            addPaths.push_back(newPath);
                    }
                }
            }
            pclose(f);
        }
        if (!addPaths.empty()) {
            char **newArgs = new char *[argsSize + 2 * addPaths.size()];
            for (int i = 0; i < argsSize; i++)
                newArgs[i] = modifiedArgs[i];
            for (int i = 0; i < (int)addPaths.size(); i++) {
                newArgs[argsSize + 2 * i] = myStrDup("-internal-isystem");
                newArgs[argsSize + 2 * i + 1] = myStrDup(addPaths[i].c_str());
                cdvmh_log(DEBUG, "Added internal system include path '%s'", addPaths[i].c_str());
            }
            argsSize = argsSize + 2 * addPaths.size();
            delete[] modifiedArgs;
            modifiedArgs = newArgs;
        }
    }

#if CLANG_VERSION_MAJOR < 4
    CompilerInvocation *invocation = new CompilerInvocation;
#else
    std::shared_ptr<CompilerInvocation> invocation(new CompilerInvocation());
#endif

#if CLANG_VERSION_MAJOR > 9
    auto tmpArgs = llvm::makeArrayRef<const char*>(modifiedArgs, modifiedArgs + argsSize);    
    clang::CompilerInvocation::CreateFromArgs(*invocation, tmpArgs, *Diagnostics);
#else
    clang::CompilerInvocation::CreateFromArgs(*invocation, modifiedArgs, modifiedArgs + argsSize, *Diagnostics);
#endif
    for (int i = 0; i < argsSize; i++)
        delete[] modifiedArgs[i];
    delete[] modifiedArgs;

    compiler = new CompilerInstance();
    compiler->setInvocation(invocation);
    compiler->createDiagnostics();
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 5
    TargetOptions *pto = new TargetOptions();
#else
    std::shared_ptr<TargetOptions> pto(new TargetOptions());
#endif
    pto->Triple = llvm::sys::getDefaultTargetTriple();
    TargetInfo *pti = TargetInfo::CreateTargetInfo(compiler->getDiagnostics(), pto); // Ownership will be transferred due to usage of reference count inside TargetInfo
    compiler->setTarget(pti); // Ownership will be transferred due to usage of reference count inside CompilerInstance
    compiler->createFileManager();
    compiler->createSourceManager(compiler->getFileManager());

    if (isDebugRemapping) {
        clang::PreprocessorOptions &PPOpts = compiler->getPreprocessorOpts();
        PPOpts.clearRemappedFiles();
        PPOpts.addRemappedFile(debugRemappingFrom, debugRemappingTo);
    }
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 5
    compiler->createPreprocessor();
#else
    compiler->createPreprocessor(clang::TU_Complete);
#endif
    preprocessor = &compiler->getPreprocessor();
	
#if CLANG_VERSION_MAJOR > 9
    preprocessor->getBuiltinInfo().initializeBuiltins(preprocessor->getIdentifierTable(), preprocessor->getLangOpts());
#elif CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 8
    preprocessor->getBuiltinInfo().InitializeBuiltins(preprocessor->getIdentifierTable(), preprocessor->getLangOpts());
#else
    preprocessor->getBuiltinInfo().initializeBuiltins(preprocessor->getIdentifierTable(), preprocessor->getLangOpts());
#endif

    compiler->createASTContext();

#if CLANG_VERSION_MAJOR > 17
    const FileEntryRef pFile = compiler->getFileManager().getFileRef(cdvFileName).get();
#elif CLANG_VERSION_MAJOR > 9
    const FileEntry *pFile = compiler->getFileManager().getFile(cdvFileName).get();
#else
    const FileEntry* pFile = compiler->getFileManager().getFile(cdvFileName);
#endif

#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 5
    compiler->getSourceManager().createMainFileID(pFile);
#else
    compiler->getSourceManager().setMainFileID(compiler->getSourceManager().createFileID(pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
#endif
}

void PassContext::parse(ASTConsumer *consumer) {
    compiler->getDiagnosticClient().BeginSourceFile(compiler->getLangOpts(), &compiler->getPreprocessor());
    clang::ParseAST(*preprocessor, consumer, compiler->getASTContext());
    compiler->getDiagnosticClient().EndSourceFile();
}

Rewriter *PassContext::getRewr(std::string name) {
    std::map<std::string, Rewriter *>::iterator it = rewriters.find(name);
    if (it != rewriters.end()) {
        return it->second;
    } else {
        Rewriter *rewr =  new Rewriter();
        rewr->setSourceMgr(compiler->getSourceManager(), compiler->getLangOpts());
        rewriters[name] = rewr;
        return rewr;
    }
}

}
