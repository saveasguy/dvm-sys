#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Basic/Version.h>
#if CLANG_VERSION_MAJOR > 10
#include <clang/Basic/FileManager.h>
#endif
#include "file_ctx.h"

using clang::ASTConsumer;
using clang::CompilerInstance;
using clang::LangOptions;
using clang::Preprocessor;
using clang::Rewriter;

namespace cdvmh {

class PassContext {
public:
    CompilerInstance *getCompiler() const { return compiler; }
    Preprocessor *getPP() const { return preprocessor; }
public:
    explicit PassContext(SourceFileContext &fileCtx): cdvFile(true) {
        cdvFileName = fileCtx.getInputFile().fileName;
        CPlusPlus = fileCtx.getInputFile().CPlusPlus;
        includeDirs = fileCtx.getProjectCtx().getOptions().includeDirs;
        addDefines = fileCtx.getProjectCtx().getOptions().addDefines;
        removeDefines = fileCtx.getProjectCtx().getOptions().removeDefines;
        isDebugRemapping = fileCtx.getProjectCtx().getOptions().dvmDebugLvl > 0 &&
                           !fileCtx.isDebugPass;
        debugRemappingFrom = fileCtx.getInputFile().fileName;
        debugRemappingTo = fileCtx.getInputFile().debugPreConvFileName;
        setup();
    }
    explicit PassContext(std::string fileName, const std::vector<std::string> &incDirs): cdvFile(false), cdvFileName(fileName), includeDirs(incDirs), isDebugRemapping(false) {
        std::string ext = toLower(getExtension(fileName));
        CPlusPlus = ext == "cpp" || ext == "hpp";
        setup();
    }
    explicit PassContext(std::string fileName): cdvFile(false), cdvFileName(fileName), isDebugRemapping(false) {
        std::string ext = toLower(getExtension(fileName));
        CPlusPlus = ext == "cpp" || ext == "hpp";
        setup();
    }
public:
    void reset() {
        clear();
        setup();
    }
    void parse(ASTConsumer *consumer);
    Rewriter *getRewr(std::string name = "");
public:
    ~PassContext() {
        clear();
    }
protected:
    void clear();
    void setup();
protected:
    bool cdvFile;
    bool CPlusPlus;
    bool isDebugRemapping;
    std::string cdvFileName;
    std::string debugRemappingFrom;
    std::string debugRemappingTo;
    std::vector<std::string> includeDirs;
    std::vector<std::pair<std::string, std::string> > addDefines;
    std::vector<std::string> removeDefines;

    CompilerInstance *compiler;
    Preprocessor *preprocessor;
    std::map<std::string, Rewriter *> rewriters;
};

}
