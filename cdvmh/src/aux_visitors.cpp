#include "aux_visitors.h"

#include <cassert>

#include <string>
#include <sstream>

namespace cdvmh {

// CollectNamesVisitor

bool CollectNamesVisitor::VisitVarDecl(VarDecl *vd) {
    if (ignoreDepth == 0)
        names.insert(vd->getName().str());
    const Type *baseType = vd->getType().getUnqualifiedType().getDesugaredType(comp.getASTContext()).split().Ty;
    if (baseType->isPointerType() || isa<IncompleteArrayType>(baseType)) {
        if (baseType->isPointerType())
            baseType = baseType->getPointeeType().getUnqualifiedType().getDesugaredType(comp.getASTContext()).split().Ty;
        else
            baseType = cast<IncompleteArrayType>(baseType)->getArrayElementTypeNoTypeQual();
    }
    while (baseType->isArrayType()) {
        baseType = baseType->getArrayElementTypeNoTypeQual();
    }
    std::string typeName = QualType(baseType, 0).getAsString();
    names.insert(typeName);
    return true;
}

bool CollectNamesVisitor::VisitTypedefDecl(TypedefDecl *td) {
    if (ignoreDepth == 0)
        names.insert(td->getName().str());
    return true;
}

bool CollectNamesVisitor::VisitFunctionDecl(FunctionDecl *f) {
    if (ignoreDepth == 0)
        if (f->getDeclName().isIdentifier())
            names.insert(f->getName().str());
    const FunctionDecl *Definition;
    bool hasBody = f->hasBody(Definition);
    bool bodyIsHere = hasBody && Definition == f;
    if (!bodyIsHere)
        ignoreDepth++;
    return true;
}

bool CollectNamesVisitor::TraverseFunctionDecl(FunctionDecl *f) {
    bool res = base::TraverseFunctionDecl(f);
    const FunctionDecl* Definition;
    bool hasBody = f->hasBody(Definition);
    bool bodyIsHere = hasBody && Definition == f;
    if (!bodyIsHere)
        ignoreDepth--;
    return res;
}

bool CollectNamesVisitor::VisitRecordDecl(RecordDecl *d) {
    if (ignoreDepth == 0)
        names.insert(d->getName().str());
    ignoreDepth++;
    return true;
}

bool CollectNamesVisitor::TraverseRecordDecl(RecordDecl *d) {
    bool res = base::TraverseRecordDecl(d);
    ignoreDepth--;
    return res;
}

bool CollectNamesVisitor::VisitDeclRefExpr(DeclRefExpr *e) {
    if (ignoreDepth == 0)
        names.insert(e->getNameInfo().getAsString());
    return true;
}

// PPDirectiveCollector

void PPDirectiveCollector::addDirective(SourceLocation beginLoc) {
    FileID fileID = comp.getSourceManager().getFileID(beginLoc);
    const char *fileText = comp.getSourceManager().getBufferData(fileID).data();
    unsigned begOffs = comp.getSourceManager().getFileOffset(beginLoc);
    while (begOffs > 0 && fileText[begOffs] != '\n')
        begOffs--;
    if (fileText[begOffs] == '\n')
        begOffs++;
    unsigned endOffs = begOffs;
    unsigned fileSize = comp.getSourceManager().getBufferData(fileID).size();
    while (endOffs < fileSize && (fileText[endOffs] != '\n' || (endOffs > 0 && fileText[endOffs - 1] == '\\')))
        endOffs++;
    endOffs--;
    SourceLocation begLoc = comp.getSourceManager().getLocForStartOfFile(fileID).getLocWithOffset(begOffs);
    SourceLocation endLoc = comp.getSourceManager().getLocForStartOfFile(fileID).getLocWithOffset(endOffs);
    directives[fileID.getHashValue()][begOffs] = SourceRange(begLoc, endLoc);
}

// IncludeCollector

void IncludeCollector::FileChanged(SourceLocation Loc, FileChangeReason Reason, SrcMgr::CharacteristicKind FileType, FileID PrevFID) {
#if CLANG_VERSION_MAJOR > 6	
    std::string prevFN = (PP.getSourceManager().getFileEntryForID(PrevFID) ? PP.getSourceManager().getFileEntryForID(PrevFID)->getName().str() : "");
#else
    std::string prevFN = (PP.getSourceManager().getFileEntryForID(PrevFID) ? PP.getSourceManager().getFileEntryForID(PrevFID)->getName() : "");
#endif

    std::string newFN = PP.getSourceManager().getFilename(Loc).str().c_str();
    if (Reason == EnterFile) {
        FileID fid = PP.getSourceManager().getFileID(Loc);
        if (ignoreLevel != 0 || newFN.empty()) {
            assert(!pendingInclusion.valid);
            ignoreLevel++;
        } else {
            bool ok = seenFIDs.insert(fid).second;
            assert(ok);
            pendingInclusion.whatFID = fid;
            commitInclusion(FileType);
        }
        cdvmh_log(TRACE, "Entered '%s' ignoreLevel=%d FileID=%u", newFN.c_str(), ignoreLevel, fid.getHashValue());
    } else if (Reason == ExitFile) {
        assert(!pendingInclusion.valid);
        assert(ignoreLevel >= 0);
        if (ignoreLevel != 0) {
            ignoreLevel--;
        } else {
            assert(activeInclusion.size() > 0);
            activeInclusion.pop_back();
        }
        cdvmh_log(TRACE, "Exited from '%s' to '%s' ignoreLevel=%d", prevFN.c_str(), newFN.c_str(), ignoreLevel);
    }
}

#if CLANG_VERSION_MAJOR < 10
void IncludeCollector::FileSkipped(const FileEntry &ParentFile, const Token &FilenameTok, SrcMgr::CharacteristicKind FileType) {
#else
void IncludeCollector::FileSkipped(const FileEntryRef & ParentFile, const Token & FilenameTok, SrcMgr::CharacteristicKind FileType) {
#endif
    if (ignoreLevel == 0) {
        pendingInclusion.isSkipped = true;
        commitInclusion(FileType);
    } else {
        assert(!pendingInclusion.valid);
    }
    cdvmh_log(TRACE, "Skipped '%s' ignoreLevel=%d", PP.getSpelling(FilenameTok).c_str(), ignoreLevel);
}

#if CLANG_VERSION_MAJOR > 15
    void IncludeCollector::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        OptionalFileEntryRef File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) {
#elif CLANG_VERSION_MAJOR > 14
    void IncludeCollector::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        Optional<FileEntryRef> File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) {
#elif CLANG_VERSION_MAJOR > 6
    void IncludeCollector::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        const FileEntry *File, StringRef SearchPath, StringRef RelativePath, const Module *Imported, SrcMgr::CharacteristicKind FileType) {
#else
    void IncludeCollector::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName, bool IsAngled, CharSourceRange FilenameRange,
        const FileEntry *File, StringRef SearchPath, StringRef RelativePath, const Module *Imported) {
#endif
    assert(!pendingInclusion.valid);
    if (ignoreLevel == 0 && File) {
        pendingInclusion = Inclusion();
        pendingInclusion.isAngled = IsAngled;
        pendingInclusion.isImport = PP.getSpelling(IncludeTok) == "import";
        pendingInclusion.isIncludeNext = PP.getSpelling(IncludeTok) == "include_next";
        pendingInclusion.spellingFN = FileName.str();
        pendingInclusion.hashLoc = HashLoc;
        pendingInclusion.where.first = PP.getSourceManager().getFileOffset(HashLoc);
        pendingInclusion.where.second = PP.getSourceManager().getFileOffset(FilenameRange.getEnd());
#if CLANG_VERSION_MAJOR > 6		
        pendingInclusion.what = getCanonicalFileName(File->getName().str());
#else
        pendingInclusion.what = getCanonicalFileName(File->getName());
#endif
        pendingInclusion.isSkipped = false;
        pendingInclusion.valid = true;
#if CLANG_VERSION_MAJOR < 4
        cdvmh_log(TRACE, "Found include '%s' angled=%d import=%d inc_next=%d spell='%s' start=%d end=%d target='%s' FileEntry='%s'",
                PP.getSpelling(IncludeTok).c_str(), (int)pendingInclusion.isAngled, (int)pendingInclusion.isImport, (int)pendingInclusion.isIncludeNext,
                pendingInclusion.spellingFN.c_str(), pendingInclusion.where.first, pendingInclusion.where.second, pendingInclusion.what.c_str(),
                File->getName());
#else
        cdvmh_log(TRACE, "Found include '%s' angled=%d import=%d inc_next=%d spell='%s' start=%d end=%d target='%s' FileEntry='%s'",
                PP.getSpelling(IncludeTok).c_str(), (int)pendingInclusion.isAngled, (int)pendingInclusion.isImport, (int)pendingInclusion.isIncludeNext,
                pendingInclusion.spellingFN.c_str(), pendingInclusion.where.first, pendingInclusion.where.second, pendingInclusion.what.c_str(),
                File->getName().data());
#endif
    } else {
#if CLANG_VERSION_MAJOR < 4
        cdvmh_log(TRACE, "Ignored include '%s' angled=%d spell='%s' FileEntry='%s'", PP.getSpelling(IncludeTok).c_str(), (int)IsAngled, FileName.str().c_str(),
                (File && File->getName() ? File->getName() : ""));
#else
        cdvmh_log(TRACE, "Ignored include '%s' angled=%d spell='%s' FileEntry='%s'", PP.getSpelling(IncludeTok).c_str(), (int)IsAngled, FileName.str().c_str(),
                (File ? File->getName().data() : ""));
#endif
    }
}

void IncludeCollector::commitInclusion(SrcMgr::CharacteristicKind FileType) {
    assert(pendingInclusion.valid);
    std::vector<Inclusion> *incs = &inclusions;
    for (int i = 0; i < (int)activeInclusion.size(); i++) {
        assert((int)incs->size() > activeInclusion[i]);
        incs = &incs->at(activeInclusion[i]).nested;
    }
    if (!pendingInclusion.isSkipped)
        activeInclusion.push_back(incs->size());
    incs->push_back(pendingInclusion);
    bool newOne = includedFiles.find(pendingInclusion.what) == includedFiles.end();
    IncludedFile &f = includedFiles[pendingInclusion.what];
    f.inclusionCount += (pendingInclusion.isSkipped == false);
    bool isSystem = (FileType == SrcMgr::C_System || FileType == SrcMgr::C_ExternCSystem);
    bool isInternal = fileCtx.isInternalInclude(pendingInclusion.spellingFN);
    if (newOne)
        f.isSystem = isSystem || isInternal;
    else
        assert(f.isSystem == isSystem);
    if (!f.isSystem && !pendingInclusion.isSkipped)
        fileCtx.addUserInclude(pendingInclusion.whatFID.getHashValue());
    cdvmh_log(TRACE, "Include is %s. Inclusion count is %d.", (isSystem ? "system" : "user"), f.inclusionCount);
    pendingInclusion = Inclusion();
}

// PragmaExpander
#if CLANG_VERSION_MAJOR > 8
void PragmaExpander::HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer, Token &PragmaTok) {
#else
void PragmaExpander::HandlePragma(Preprocessor & PP, PragmaIntroducerKind Introducer, Token & PragmaTok) {
#endif
    std::string lastDirective;
    int linesSinceDirective = 0;
    {
        std::stringstream ss(OS.str());
        while (!ss.eof()) {
            std::string ln;
            std::getline(ss, ln);
            if (!strncmp(ln.c_str(), "#line ", 6) || !strncmp(ln.c_str(), "# ", 2)) {
                lastDirective = ln;
                linesSinceDirective = 0;
            } else {
                linesSinceDirective++;
            }
        }
    }
    OS << '\n';
    PresumedLoc ploc = PP.getSourceManager().getPresumedLoc(PragmaTok.getLocation());
    if (ploc.isValid()) {
        OS << "#line " << ploc.getLine() << " \"";
        OS.write_escaped(ploc.getFilename());
        OS << "\"\n";
    }
    OS << "#pragma";
    while (PragmaTok.isNot(tok::eod)) {
        OS << ' ';
        std::string TokSpell = PP.getSpelling(PragmaTok);
        OS << TokSpell;
        PP.Lex(PragmaTok);
    }
    OS << '\n';
    if (!lastDirective.empty()) {
        std::string::size_type p1 = lastDirective.find('"');
        std::string::size_type p2 = lastDirective.rfind('"');
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::string fn = lastDirective.substr(p1, p2 - p1 + 1);
            std::stringstream ss(lastDirective);
            std::string ignore;
            int lineNumber = -1;
            ss >> ignore >> lineNumber;
            if (lineNumber > 0)
                OS << "#line " << (lineNumber + linesSinceDirective - 1) << " " << fn << '\n';
        }
    }
}

// IncludeExpanderAndRewriter

void IncludeExpanderAndRewriter::processInclusions(FileID parentFID, const std::vector<Inclusion> &incs, bool &isChanged, bool &isHard) {
    isChanged = false;
    isHard = false;
    for (int i = 0; i < (int)incs.size(); i++) {
        bool thisIsChanged, thisIsHard;
        processInclusion(incs[i], thisIsChanged, thisIsHard);
        if (!thisIsChanged) {
            // No changes to contents => leave the file included unchanged and at its original place.
            // Check accessibility with the original inclusion directive
            if (!incs[i].isAngled) {
                // Angled is inclusion-source-independent
                if (getCanonicalFileName(incs[i].spellingFN, getDirectory(mainOutFN)) != incs[i].what) {
                    // Is unreachable relative to our directory
                    cdvmh_log(TRACE, "Can't reach '%s' from '%s' by '%s'", incs[i].what.c_str(), mainOutFN.c_str(), incs[i].spellingFN.c_str());
                    const std::vector<std::string> &incDirs = projectCtx.getOptions().includeDirs;
                    bool ok = false;
                    for (int j = 0; j < (int)incDirs.size(); j++) {
                        std::string candidateFN = getCanonicalFileName(incs[i].spellingFN, incDirs[j]);
                        if (candidateFN == incs[i].what) {
                            ok = true;
                            break;
                        } else if (fileExists(candidateFN)) {
                            // Wrong file will be included if we do not alter the directive
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) {
                        // Is unreachable relative to all user-specified include directories => rewrite include directive
                        std::string target = incs[i].what;
                        // If we were given this file also, prefer to use a converted version
                        if (projectCtx.hasInputFile(incs[i].what, true))
                            target = projectCtx.getInputFile(incs[i].what, true).canonicalConverted;
                        std::string toInsert = "#include \"" + getShortestPath(getDirectory(mainOutFN), target, '/') + "\"";
                        rewr.ReplaceText(incs[i].hashLoc, incs[i].where.second - incs[i].where.first, toInsert);
                        isChanged = true;
                    }
                }
            }
        } else if (thisIsHard) {
            isHard = true;
            const RewriteBuffer *rewriteBuf = rewr.getRewriteBufferFor(incs[i].whatFID);
            assert(rewriteBuf);
            std::string toInsert(rewriteBuf->begin(), rewriteBuf->end());
            rewr.ReplaceText(incs[i].hashLoc, incs[i].where.second - incs[i].where.first, toInsert);
            isChanged = true;
        } else {
            // Have changes, but it is not mandatory to expand it
            if (projectCtx.hasInputFile(incs[i].what, true)) {
                std:: string target = projectCtx.getInputFile(incs[i].what, true).canonicalConverted;
                std::string toInsert = "#include \"" + getShortestPath(getDirectory(mainOutFN), target, '/') + "\"";
                rewr.ReplaceText(incs[i].hashLoc, incs[i].where.second - incs[i].where.first, toInsert);
            } else {
                const RewriteBuffer *rewriteBuf = rewr.getRewriteBufferFor(incs[i].whatFID);
                assert(rewriteBuf);
                std::string toInsert(rewriteBuf->begin(), rewriteBuf->end());
                rewr.ReplaceText(incs[i].hashLoc, incs[i].where.second - incs[i].where.first, toInsert);
            }
            isChanged = true;
        }
    }
}
void IncludeExpanderAndRewriter::processInclusion(const Inclusion &inc, bool &isChanged, bool &isHard) {
    isChanged = false;
    isHard = false;
    if (fileCtx.isUserInclude(inc.whatFID.getHashValue())) {
        processInclusions(inc.whatFID, inc.nested, isChanged, isHard);
        if (rewr.getRewriteBufferFor(inc.whatFID))
            isChanged = true;
        if (fileCtx.isExpansionForced(inc.whatFID.getHashValue()))
            isHard = true;
    }
}

}
