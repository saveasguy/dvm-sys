#include "converter.h"

#include "messages.h"

namespace cdvmh {

bool isGlobalC(const Decl *d) {
    const DeclContext *dc = d->getDeclContext();
    return !isa<ParmVarDecl>(d) && (dc->isTranslationUnit() || dc->isExternCContext());
}

std::string genRtType(std::string baseType) {
    std::string result;
    if (baseType == "int") {
        result = "rt_INT";
    } else if (baseType == "bool" || baseType == "_Bool") {
        result = "rt_LOGICAL";
    } else if (baseType == "long") {
        result = "rt_LONG";
    } else if (baseType == "long long") {
        result = "rt_LLONG";
    } else if (baseType == "float") {
        result = "rt_FLOAT";
    } else if (baseType == "double") {
        result = "rt_DOUBLE";
    } else if (baseType == "_Complex float") {
        result = "rt_FLOAT_COMPLEX";
    } else if (baseType == "_Complex double") {
        result = "rt_DOUBLE_COMPLEX";
    } else {
        result = "rt_UNKNOWN";
    }
    return result;
}

std::string declToStr(Decl *d, bool addStatic, bool addExtern, bool truncateBody) {
    VarDecl *vd = llvm::dyn_cast<VarDecl>(d);
    FunctionDecl *fd = llvm::dyn_cast<FunctionDecl>(d);
    Stmt *bodySave = 0;
    StorageClass prevClass = SC_None;
    if (truncateBody) {
        if (vd && vd->hasInit()) {
            bodySave = *vd->getInitAddress();
            *vd->getInitAddress() = 0;
        } else if (fd && fd->doesThisDeclarationHaveABody()) {
            bodySave = fd->getBody();
            fd->setLazyBody(0);
        }
    }
    if (addStatic || addExtern) {
        if (vd)
            prevClass = vd->getStorageClass();
        else if (fd)
            prevClass = fd->getStorageClass();
    }
    std::string ss;
    llvm::raw_string_ostream ost(ss);
    if (addStatic && (vd || fd) && prevClass != SC_Static)
        ost << "static ";
    else if (addExtern && vd && prevClass != SC_Extern)
        ost << "extern ";
    d->print(ost);
    if (vd && bodySave)
        *vd->getInitAddress() = bodySave;
    else if (fd &&bodySave)
        fd->setBody(bodySave);
    return ost.str();
}

std::string convertToString(Stmt *s, SourceManager &srcMgr, const LangOptions &langOpts, bool preserveSourceText) {
    if (preserveSourceText)
       return Lexer::getSourceText(CharSourceRange::getTokenRange(s->getSourceRange()), srcMgr, langOpts).str();
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 6
    Rewriter rewr(srcMgr, langOpts);
    return rewr.ConvertToString(s);
#else
    std::string SStr;
    llvm::raw_string_ostream S(SStr);
    s->printPretty(S, 0, PrintingPolicy(langOpts));
    return S.str();
#endif
}

std::string convertToString(Stmt *s, CompilerInstance &comp, bool preserveSourceText) {
    return convertToString(s, comp.getSourceManager(), comp.getLangOpts(), preserveSourceText);
}

VarState *fillVarState(const VarDecl *vd, bool CPlusPlus, CompilerInstance &comp, VarState *varState) {
    SourceManager &srcMgr = comp.getSourceManager();
    SourceLocation fileLoc = srcMgr.getFileLoc(vd->getLocation());
    std::string fileName = srcMgr.getFilename(fileLoc).str();
    FileID fileID = srcMgr.getFileID(fileLoc);
    int line = srcMgr.getLineNumber(fileID, srcMgr.getFileOffset(fileLoc));
    std::string varName = vd->getName().str();
    bool hasRestrict = vd->getType().isRestrictQualified();
    const Type *baseType = vd->getType().getUnqualifiedType().getDesugaredType(comp.getASTContext()).split().Ty;
    std::vector<MyExpr> sizes;
    if (baseType->isPointerType() || isa<IncompleteArrayType>(baseType)) {
        sizes.push_back(MyExpr());
        if (baseType->isPointerType())
            baseType = baseType->getPointeeType().getUnqualifiedType().getDesugaredType(comp.getASTContext()).split().Ty;
        else
            baseType = cast<IncompleteArrayType>(baseType)->getArrayElementTypeNoTypeQual();
        cdvmhLog(DONT_LOG, fileName, line, "Outer pointer/incomplete array type found");
    }
    while (baseType->isArrayType()) {
        //checkUserErr(baseType->getAsArrayTypeUnsafe()->getSizeModifier() == ArrayType::Normal, fileName, line,
        //        "That kind of array size modifier is not supported for variable '%s'", varName.c_str());
        MyExpr nextSize;
        if (const ConstantArrayType *ca = llvm::dyn_cast<const ConstantArrayType>(baseType)) {
            nextSize.strExpr = toStr(ca->getSize().getZExtValue());
        } else if (const VariableArrayType *va = llvm::dyn_cast<const VariableArrayType>(baseType)) {
            Expr *e = va->getSizeExpr();
            // TODO: Fill somehow info on references in this expression
            nextSize.strExpr = convertToString(e, comp);
        } else if (const DependentSizedArrayType *dsa = llvm::dyn_cast<const DependentSizedArrayType>(baseType)) {
            nextSize.strExpr = convertToString(dsa->getSizeExpr(), comp);
        } else {
            //checkUserErr(false, fileName, line,
            //        "That kind of array type is not supported for variable '%s'", varName.c_str());
            cdvmh_log(WARNING, 52, MSG(52), baseType->getTypeClassName());
            nextSize.strExpr = "0";
        }
        sizes.push_back(nextSize);
        baseType = baseType->getArrayElementTypeNoTypeQual();
    }
    std::string typeName = QualType(baseType, 0).getAsString();
    // XXX: dirty
    if (typeName == "_Bool" && CPlusPlus)
        typeName = "bool";
    if (!varState)
        varState = new VarState;
    varState->init(varName, typeName, sizes);
    if (strstr(baseType->getCanonicalTypeInternal().getAsString().c_str(), "type-parameter-"))
        varState->hasDependentBaseType = true;
    if (hasRestrict)
        varState->canBeRestrict = true;
    return varState;
}

// MyDeclContext

VarDecl *MyDeclContext::lookupVar(const std::string &varName) const {
    std::map<std::string, VarDecl *>::const_iterator it = vars.find(varName);
    if (it != vars.end())
        return it->second;
    else if (parent)
        return parent->lookupVar(varName);
    else
        return 0;
}

bool MyDeclContext::add(VarDecl *vd) {
    bool res = false;
    std::string varName = vd->getName().str();
    std::string varFullName = vd->getQualifiedNameAsString();
    if (!varName.empty() && varName == varFullName) {
        std::map<std::string, VarDecl *>::iterator it = vars.find(varName);
        if (it != vars.end()) {
            VarDecl *def1, *def2;
            def1 = vd->getDefinition();
            def2 = it->second->getDefinition();
            checkIntErrN(def1 == def2, 91, varName.c_str(), (void *)def1, (void *)def2);
            res = vd != it->second;
        } else {
            res = true;
        }
        vars[varName] = vd;
    }
    return res;
}

// ConverterASTVisitor

ConverterASTVisitor::ConverterASTVisitor(SourceFileContext &aFileCtx, CompilerInstance &aComp, Rewriter &R): fileCtx(aFileCtx),
        projectCtx(fileCtx.getProjectCtx()), opts(projectCtx.getOptions()), comp(aComp), rewr(R), srcMgr(rewr.getSourceMgr()), langOpts(rewr.getLangOpts()) {
    indentStep = genIndent(1, fileCtx.useTabs());
    inRegion = false;
    regionStmt = 0;
    curRegion = "0";
    inParLoop = false;
    parLoopStmt = 0;
    inParLoopBody = false;
    parLoopBodyStmt = 0;
    parallelRmaDesc = 0;
    inHostSection = false;
    hostSectionStmt = 0;
    declContexts.push_back(new MyDeclContext());
    preventExpandToDelete = false;
}

void ConverterASTVisitor::addToDeclGroup(Decl* head, Decl* what) {
    declGroups[head].push_back(what);
}

void ConverterASTVisitor::addSeenDecl(Decl *d) {
    if (declOrder.find(d) == declOrder.end()) {
        int index = declOrder.size();
        declOrder[d] = index;
    } else {
        PresumedLoc ploc = srcMgr.getPresumedLoc(srcMgr.getFileLoc(d->getLocation()));
        cdvmh_log(ERROR, "Duplicate of decl in %s:%u:%u", ploc.getFilename(), ploc.getLine(), ploc.getColumn());
    }
}

std::string ConverterASTVisitor::declToStrForBlank(bool shallow, Decl *d) {
    VarDecl *vd = llvm::dyn_cast<VarDecl>(d);
    FunctionDecl *fd = llvm::dyn_cast<FunctionDecl>(d);
    bool addStatic = false;
    bool addExtern = false;
    bool truncateBody = false;
    if (fd) {
        bool hasBody = fd->doesThisDeclarationHaveABody();
        bool isStatic = fd->getFirstDecl()->getStorageClass() == SC_Static;
        if (shallow && !isStatic) {
            truncateBody = true;
        }
        if (hasBody && !truncateBody) {
            addStatic = true;
        }
    }
    if (vd) {
        bool isStatic = vd->getStorageClass() == SC_Static;
        PresumedLoc ploc = srcMgr.getPresumedLoc(vd->getLocation());
        checkUserErr(!isStatic, ploc.getFilename(), ploc.getLine(), "Static variable usage is forbidden inside regions and parallel loops");
        if (shallow) {
            truncateBody = true;
            addExtern = true;
        } else {
            addStatic = true;
        }
    }
    return declToStr(d, addStatic, addExtern, truncateBody);
}

void ConverterASTVisitor::afterTraversing() {
    if (!opts.useBlank && !addedCudaFuncs.empty()) {
        for (std::set<const FunctionDecl *>::iterator it = addedCudaFuncs.begin(); it != addedCudaFuncs.end(); it++) {
            fileCtx.addToCudaHeading(std::string("__device__ ") + declToStr(const_cast<FunctionDecl *>(*it), true, false, false));
        }
    }
    std::map<int, Decl *> orderedDecls;
    for (std::set<Decl *>::const_iterator it = blankHandlerDeclsDeep.begin(); it != blankHandlerDeclsDeep.end(); it++) {
        Decl *d = *it;
        orderedDecls[declOrder[d]] = d;
    }
    for (std::set<Decl *>::const_iterator it = blankHandlerDeclsSystem.begin(); it != blankHandlerDeclsSystem.end(); it++) {
        Decl *d = *it;
        orderedDecls[declOrder[d]] = d;
    }
    std::string blankHeading;
    for (std::map<int, Decl *>::iterator it = orderedDecls.begin(); it != orderedDecls.end(); it++) {
        Decl *d = it->second;
        bool system = blankHandlerDeclsSystem.find(d) != blankHandlerDeclsSystem.end();
        bool shallow = blankHandlerDeclsShallow.find(d) != blankHandlerDeclsShallow.end();
        if (system) {
            FileID includeID = srcMgr.getFileID(srcMgr.getExpansionLoc(d->getLocation()));
            for (;;) {
                SourceLocation includeLoc = srcMgr.getIncludeLoc(includeID);
                bool isSystem = srcMgr.getFileCharacteristic(includeLoc) == SrcMgr::C_System || srcMgr.getFileCharacteristic(includeLoc) == SrcMgr::C_ExternCSystem;
                if (!isSystem)
                  break;
                includeID = srcMgr.getFileID(includeLoc);
            }
            const FileEntry *file = srcMgr.getFileEntryForID(includeID);
            std::string fileToInclude = getBaseName(llvm::StringRef(file->getName()).str());
            blankHeading += "#include <" + fileToInclude + ">";
        } else if (!shallow) {
            blankHeading += "#ifdef CDVMH_FOR_OFFLOAD\n";
            blankHeading += declToStrForBlank(false, d) + ";\n";
            blankHeading += "#endif\n";
        } else {
            std::string shallowStr = declToStrForBlank(true, d);
            std::string deepStr = declToStrForBlank(false, d);
            if (shallowStr == deepStr) {
                blankHeading += shallowStr + ";\n";
            } else {
                blankHeading += "#ifdef CDVMH_FOR_OFFLOAD\n";
                blankHeading += deepStr + ";\n";
                blankHeading += "#else\n";
                blankHeading += shallowStr + ";\n";
                blankHeading += "#endif\n";
            }
        }
        blankHeading += "\n";
    }
    if (!blankHeading.empty()) {
        blankHeading += "\n";
        fileCtx.addToBlankHeading(blankHeading);
    }
}

void ConverterASTVisitor::genUnbinded(FileID fileID, int line) {
    int prevPragmaLine;
    do {
        prevPragmaLine = fileCtx.getLatestPragmaLine(fileID.getHashValue());
        genActuals(fileID, line);
        genRedistributes(fileID, line);
        genRealignes(fileID, line);
        genIntervals(fileID, line);
        genShadowAdds(fileID, line);
        genLocalizes(fileID, line);
        genUnlocalizes(fileID, line);
        genArrayCopies(fileID, line);
    } while (fileCtx.getLatestPragmaLine(fileID.getHashValue()) > prevPragmaLine);
}

void ConverterASTVisitor::checkNonDvmExpr(const MyExpr &expr, DvmPragma *curPragma) {
    for (std::set<std::string>::iterator it = expr.usedNames.begin(); it != expr.usedNames.end(); it++) {
        VarDecl *vd = seekVarDecl(*it);
        if (vd) {
            checkIntErrN(varStates.find(vd) != varStates.end(), 92, vd->getNameAsString().c_str());
            VarState *varState = &varStates[vd];
            checkDirErrN(varState->isRegular, 401, expr.strExpr.c_str());
        }
    }
}

bool ConverterASTVisitor::flushToDelete(std::string &toInsert, std::string indent, int uptoLevel) {
    if (uptoLevel < 0)
        uptoLevel = (int)toDelete.size() - 1;
    bool haveSomething = false;
    for (int i = (int)toDelete.size() - 1; i >= 0 && i >= uptoLevel; i--) {
        for (int j = (int)toDelete[i].size() - 1; j >= 0; j--) {
            toInsert += indent + toDelete[i][j].first + "(" + toDelete[i][j].second + ");\n";
            haveSomething = true;
        }
    }
    return haveSomething;
}

bool ConverterASTVisitor::isFull(Stmt *s) {
    std::string str = convertToString(s);
    if (!str.empty() && str[str.size() - 1] == '\n')
        str.resize(str.size() - 1);
    if (str.empty())
        return false;
    else if (str[str.size() - 1] == '}' || str[str.size() - 1] == ';')
        return true;
    else
        return false;
}

bool ConverterASTVisitor::isCudaFriendly(FunctionDecl *f) {
    // TODO: Implement
    if (!f)
        return false;
    return f->isDefined();
}

class FuncAdder: public RecursiveASTVisitor<FuncAdder> {
public:
    FuncAdder(ConverterASTVisitor *converter): conv(converter) {}
public:
    bool VisitDeclRefExpr(DeclRefExpr *e) {
        std::string name = e->getNameInfo().getAsString();
        bool globalDecl = isGlobalC(e->getDecl());
        if (isa<FunctionDecl>(e->getDecl())) {
            if (globalDecl && !conv->projectCtx.hasCudaReplacement(name)) {
                conv->addFuncForCuda(cast<FunctionDecl>(e->getDecl()));
            }
        }
        return true;
    }
protected:
    ConverterASTVisitor *conv;
};

bool ConverterASTVisitor::addFuncForCuda(FunctionDecl *f) {
    if (!isCudaFriendly(f))
        return false;
    const FunctionDecl *def;
    f->isDefined(def);
    if (addedCudaFuncs.find(def) != addedCudaFuncs.end() || addCudaFuncs.find(def) != addCudaFuncs.end())
        return true;
    addCudaFuncs.insert(def);
    FuncAdder adder(this);
    adder.TraverseStmt(f->getBody());
    return true;
}

SourceLocation ConverterASTVisitor::escapeMacroBegin(SourceLocation loc) {
    bool ok = true;
    while (loc.isMacroID()) {
        ok = Lexer::isAtStartOfMacroExpansion(loc, srcMgr, langOpts, &loc);
        if (!ok)
            break;
    }
    if (!ok) {
        PresumedLoc ploc = srcMgr.getPresumedLoc(srcMgr.getFileLoc(loc));
        userErrN(ploc.getFilename(), ploc.getLine(), 53);
    }
    return loc;
}

SourceLocation ConverterASTVisitor::escapeMacroEnd(SourceLocation loc) {
    bool ok = true;
    while (loc.isMacroID()) {
        ok = Lexer::isAtEndOfMacroExpansion(loc, srcMgr, langOpts, &loc);
        if (!ok)
            break;
    }
    if (!ok) {
        PresumedLoc ploc = srcMgr.getPresumedLoc(srcMgr.getFileLoc(loc));
        userErrN(ploc.getFilename(), ploc.getLine(), 53);
    }
    return loc;
}

VarDecl *ConverterASTVisitor::seekVarDecl(std::string name, MyDeclContext *context) {
    if (!context) {
        checkIntErrN(!declContexts.empty(), 94);
        context = declContexts.back();
    }
    return context->lookupVar(name);
}

bool ConverterASTVisitor::VisitDecl(Decl *d) {
    SourceLocation fileLoc = srcMgr.getFileLoc(d->getLocation());
    std::string fileName = srcMgr.getFilename(fileLoc).str();
    FileID fileID = srcMgr.getFileID(fileLoc);
    int line = srcMgr.getLineNumber(fileID, srcMgr.getFileOffset(fileLoc));
    if (!isDeclAllowed() && !(isa<VarDecl>(d) && outerPrivates.find(cast<VarDecl>(d)) != outerPrivates.end()))
        userErrN(fileName, line, 402);
    addSeenDecl(d);
    return true;
}

bool ConverterASTVisitor::VisitTagDecl(TagDecl* td) {
    if (opts.enableTags) {
        if (!td->getDeclContext()->isFileContext())
            return true;
        if (!td->isStruct() && !td->isClass() && !td->isUnion() && !td->isEnum())
            return true;
        if (td->getLocation().isInvalid() || srcMgr.getFileCharacteristic(td->getLocation()) != clang::SrcMgr::C_User)
            return true;
        if (RecordDecl *rd = dyn_cast<RecordDecl>(td)) {
            for (DeclContext::decl_iterator i = rd->decls_begin(), ei = rd->decls_end(); i != ei; ++i) {
                if (FieldDecl *fd = dyn_cast<FieldDecl>(*i)) {
                    QualType qt = fd->getType().getUnqualifiedType();
                    for (const Type *t = qt.getTypePtrOrNull(); t && (isa<PointerType>(t) || isa<ConstantArrayType>(t)); t = qt.getTypePtrOrNull())
                        if (isa<PointerType>(t))
                            qt = t->getPointeeType().getUnqualifiedType();
                        else
                            qt = cast<ArrayType>(t)->getElementType().getUnqualifiedType();
                    std::string typeName = qt.getAsString();
                    if (!fileCtx.hasCudaGlobalDecl(typeName) && fileCtx.getProjectCtx().getCudaReplacement(typeName) != typeName)
                        return true;
                } else {
                    return true;
                }
            }
        } else if (EnumDecl *ed = dyn_cast<EnumDecl>(td)) {
            if (!ed->isScoped())
                for (EnumDecl::enumerator_iterator i = ed->enumerator_begin(), ei = ed->enumerator_end(); i != ei; ++i)
                    fileCtx.addCudaGlobalDecl(i->getName().str(), rewr.getRewrittenText(i->getSourceRange()), false);
        }
        std::string declString;
        llvm::raw_string_ostream os(declString);
        td->print(os, PrintingPolicy(langOpts));
        os << ";";
        if (td->isCompleteDefinition()) {
            std::string typeName = td->getASTContext().getTypeDeclType(td).getAsString();
            fileCtx.addCudaGlobalDecl(typeName, os.str());
        } else {
            fileCtx.addCudaGlobalDecl(os.str());
        }
    }
  return true;
}

bool ConverterASTVisitor::VisitTypedefNameDecl(TypedefNameDecl *td) {
    if (opts.enableTags) {
        if (!td->getDeclContext()->isFileContext())
            return true;
        if (td->getLocation().isInvalid() || srcMgr.getFileCharacteristic(td->getLocation()) != clang::SrcMgr::C_User)
            return true;
        QualType qt = td->getUnderlyingType().getUnqualifiedType();
        for (const Type *t = qt.getTypePtrOrNull(); t && (isa<PointerType>(t) || isa<ConstantArrayType>(t)); t = qt.getTypePtrOrNull())
            if (isa<PointerType>(t))
                qt = t->getPointeeType().getUnqualifiedType();
            else
                qt = cast<ArrayType>(t)->getElementType().getUnqualifiedType();
        std::string typeName = qt.getAsString();
        if (!fileCtx.hasCudaGlobalDecl(typeName) && fileCtx.getProjectCtx().getCudaReplacement(typeName) != typeName)
            return true;
        std::string declString;
        llvm::raw_string_ostream os(declString);
        td->print(os, PrintingPolicy(langOpts));
        os << ";";
        fileCtx.addCudaGlobalDecl(td->getName().str(), os.str());
    }
    return true;
}

VarState ConverterASTVisitor::fillVarState(VarDecl *vd) {
    SourceLocation fileLoc = srcMgr.getFileLoc(vd->getLocation());
    std::string fileName = srcMgr.getFilename(fileLoc).str();
    FileID fileID = srcMgr.getFileID(fileLoc);
    int line = srcMgr.getLineNumber(fileID, srcMgr.getFileOffset(fileLoc));
    std::string varName = vd->getName().str();
    checkIntErrN(varStates.find(vd) == varStates.end(), 95, varName.c_str(), fileName.c_str(), line);
    VarState varState;
    cdvmh::fillVarState(vd, fileCtx.getInputFile().CPlusPlus, comp, &varState);
    return varState;
}

bool ConverterASTVisitor::VisitDeclStmt(DeclStmt *ds) {
    SourceLocation fileLoc = srcMgr.getFileLoc(ds->getLocEnd());
    std::string fileName = srcMgr.getFilename(fileLoc).str();
    FileID fileID = srcMgr.getFileID(fileLoc);
    int line = srcMgr.getLineNumber(fileID, srcMgr.getFileOffset(fileLoc));
    if (projectCtx.hasInputFile(fileName))
        cdvmhLog(TRACE, fileName, line, "DeclStmt: isSingle=%d", (int)ds->isSingleDecl());
    if (ds->isSingleDecl()) {
        addToDeclGroup(ds->getSingleDecl(), ds->getSingleDecl());
    } else {
        for (DeclGroupRef::iterator it = ds->getDeclGroup().begin(); it != ds->getDeclGroup().end(); it++)
            addToDeclGroup(*ds->getDeclGroup().begin(), *it);
    }
    return true;
}

bool ConverterASTVisitor::TraverseFunctionProtoTypeLoc(FunctionProtoTypeLoc ft) {
    SourceLocation fileLoc = srcMgr.getFileLoc(ft.getLocStart());
    PresumedLoc ploc = srcMgr.getPresumedLoc(fileLoc);
    cdvmhLog(DONT_LOG, ploc.getFilename(), ploc.getLine(), "Entering MyDeclContext(Function)");
    enterDeclContext(true);
    bool res = base::TraverseFunctionProtoTypeLoc(ft);
    checkIntervalBalance(fileLoc);
    leaveDeclContext();
    cdvmhLog(DONT_LOG, ploc.getFilename(), ploc.getLine(), "Leaving MyDeclContext(Function)");
    return res;
}

bool ConverterASTVisitor::VisitCompoundStmt(CompoundStmt *s) {
    if (preventExpandToDelete)
        preventExpandToDelete = false;
    else
        toDelete.resize(toDelete.size() + 1);
    enterDeclContext(true);
    if (FunctionDecl *f = findUpwards<FunctionDecl>(s, 1)) {
        for (int i = 0; i < (int)f->getNumParams(); i++)
            declContexts.back()->add(f->getParamDecl(i));
    }
    return true;
}

// DeclUsageCollector

bool DeclUsageCollector::VisitDeclRefExpr(DeclRefExpr *e) {
    if (isa<FunctionDecl>(e->getDecl())) {
        FunctionDecl *fd = cast<FunctionDecl>(e->getDecl());
        FunctionDecl *first = fd->getFirstDecl();
        addDecl(first);
        const FunctionDecl *definition = 0;
        if (first->hasBody(definition) && definition != first) {
            addDecl(const_cast<FunctionDecl *>(definition), first->isExternallyVisible());
        }
    } else if (isa<VarDecl>(e->getDecl())) {
        VarDecl *vd = cast<VarDecl>(e->getDecl());
        TraverseType(vd->getType());
    } else if (isa<EnumConstantDecl>(e->getDecl())) {
        EnumConstantDecl *cd = cast<EnumConstantDecl>(e->getDecl());
        DeclContext *dc = cd->getDeclContext();
        EnumDecl *ed = cast<EnumDecl>(dc);
        if (ed->getDefinition())
            ed = ed->getDefinition();
        if (ed) {
            addDecl(ed);
        }
    }
    return true;
}

bool DeclUsageCollector::TraverseType(QualType t) {
    addType(t.getTypePtrOrNull());
    return true;
}

bool DeclUsageCollector::VisitType(Type *t) {
    addType(t);
    return true;
}

bool DeclUsageCollector::VisitFunctionDecl(FunctionDecl *fd) {
    if (fd->hasBody() && !deepMode && fd->isExternallyVisible()) {
        becomeDeepInside = fd->getBody();
    }
    return true;
}

bool DeclUsageCollector::TraverseStmt(Stmt *s) {
    if (s == becomeDeepInside) {
        deepMode = true;
    }
    bool res = base::TraverseStmt(s);
    if (s == becomeDeepInside) {
        deepMode = false;
        becomeDeepInside = 0;
    }
    return res;
}

void DeclUsageCollector::addType(const Type *t) {
    if (!t)
        return;
    Decl *d = 0;
    if (t->isRecordType()) {
        d = t->getAs<RecordType>()->getDecl();
    } else if (t->isEnumeralType()) {
        d = t->getAs<EnumType>()->getDecl();
    } else if (t->isArrayType()) {
        addType(t->getAsArrayTypeUnsafe()->getElementType().getTypePtrOrNull());
    } else if (isa<TypedefType>(t)) {
        d = t->getAs<TypedefType>()->getDecl();
    }
    if (d) {
        addDecl(d);
    }
}

void DeclUsageCollector::addDecl(Decl *d, bool forceDeep) {
    bool asDeep = deepMode || forceDeep;
    if (referencedDeclsDeep.find(d) == referencedDeclsDeep.end() || (!asDeep && referencedDeclsShallow.find(d) == referencedDeclsShallow.end())) {
        addAsReferenced(d, asDeep);
        DeclUsageCollector collector(referencedDeclsShallow, referencedDeclsDeep, asDeep);
        collector.TraverseDecl(d);
        for (std::set<Decl *>::const_iterator it = collector.getReferencedDeclsShallow().begin(); it != collector.getReferencedDeclsShallow().end(); it++) {
            referencedDeclsShallow.insert(*it);
        }
        for (std::set<Decl *>::const_iterator it = collector.getReferencedDeclsDeep().begin(); it != collector.getReferencedDeclsDeep().end(); it++) {
            referencedDeclsDeep.insert(*it);
        }
    }
}

}
