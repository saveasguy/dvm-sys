#include "pragma_parser.h"
#if CLANG_VERSION_MAJOR > 8
#include "external_preprocessor.h"
#endif

#include <cstdio>

#include "messages.h"

using namespace clang;

namespace cdvmh {

#define COLUMN (int)comp.getSourceManager().getColumnNumber(fileID, comp.getSourceManager().getFileOffset(Tok.getLocation()))

#if CLANG_VERSION_MAJOR > 8
void DvmPragmaHandler::HandlePragma(Preprocessor &BasicPP, PragmaIntroducer Introducer, Token &FirstToken) {
    SourceLocation loc = FirstToken.getLocation();
    SmallVector<Token, 64> tokensToRelex;
    do {
        BasicPP.LexUnexpandedToken(FirstToken);
        tokensToRelex.push_back(FirstToken);
    } while (!FirstToken.is(tok::eod));
    ExternalPreprocessor PP(BasicPP, tokensToRelex);
#else
void DvmPragmaHandler::HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer, Token & FirstToken) {
    SourceLocation loc = FirstToken.getLocation();
#endif
    loc = rewr.getSourceMgr().getFileLoc(loc);
    std::string srcFileName = comp.getSourceManager().getFilename(loc).str();
    fileID = rewr.getSourceMgr().getFileID(loc);
    int srcLine = comp.getSourceManager().getLineNumber(fileID, comp.getSourceManager().getFileOffset(loc));
    int column = comp.getSourceManager().getColumnNumber(fileID, comp.getSourceManager().getFileOffset(loc));
    //checkUserErrN(projectCtx.hasInputFile(srcFileName), srcFileName, srcLine, 23);
    PresumedLoc ploc = rewr.getSourceMgr().getPresumedLoc(loc);
    std::string fileName(ploc.getFilename());
    int line = ploc.getLine();
    SourceLocation endLoc;
    {
        Token tmpTok;
        PP.EnableBacktrackAtThisPos();
        do {
            PP.LexNonComment(tmpTok);
        } while (tmpTok.isNot(tok::eod));
        PP.Backtrack();
        endLoc = tmpTok.getLocation();
    }
    int srcLineSpan = comp.getSourceManager().getLineNumber(fileID, comp.getSourceManager().getFileOffset(endLoc)) - srcLine + 1;
    const char *beg = comp.getSourceManager().getBufferData(fileID).data() + comp.getSourceManager().getFileOffset(loc);
    const char *end = comp.getSourceManager().getBufferData(fileID).data() + comp.getSourceManager().getFileOffset(endLoc);
    char *buf = myStrDup(beg, end - beg);
    cdvmhLog(TRACE, fileName, line, "Pragma at column %d, line span = %d, raw data = '%s'\n", column, srcLineSpan, buf);
    delete[] buf;

    // Remove or turn off our pragmas
    if (!fileCtx.isDebugPass) {
        if (!projectCtx.getOptions().savePragmas) {
            SourceLocation lineBegLoc = comp.getSourceManager().translateLineCol(fileID, srcLine, 1);
            const char *lineBeg = comp.getSourceManager().getBufferData(fileID).data() + comp.getSourceManager().getFileOffset(lineBegLoc);
            int newLineChars = 0;
            if (end[newLineChars] == '\r')
                newLineChars++;
            if (end[newLineChars] == '\n')
                newLineChars++;
            rewr.RemoveText(lineBegLoc, end - lineBeg + newLineChars);
        } else {
            for (int i = 0; i < srcLineSpan; i++) {
                SourceLocation lineBegLoc = comp.getSourceManager().translateLineCol(fileID, srcLine + i, 1);
                rewr.InsertText(lineBegLoc, "// ", true, false);
            }
        }
    }

    curPragma = new DvmPragma(DvmPragma::pkNoKind);
    curPragma->fileName = fileName;
    curPragma->line = line;
    curPragma->srcFileName = srcFileName;
    curPragma->srcLine = srcLine;
    curPragma->srcLineSpan = srcLineSpan;
    Token Tok;
    PP.LexNonComment(Tok);
    checkDirErrN(Tok.isAnyIdentifier() || Tok.is(tok::kw_template), 302);
    std::string tokStr = Tok.getIdentifierInfo()->getName().str();
    if (tokStr == "array") {
        // Distributed array
        PragmaDistribArray *curPragma = new PragmaDistribArray();
        curPragma->copyCommonInfo(this->curPragma);
        curPragma->dynamicFlag = 0;
        PP.LexNonComment(Tok);
        if (!Tok.is(tok::eod)) {
            while (Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                if (tokStr == "distribute") {
                    checkDirErrN(curPragma->alignFlag == -1, 312);
                    curPragma->alignFlag = 0;
                    PP.LexNonComment(Tok);
                    curPragma->distribRule = parseDistribRule(PP, Tok);
                    int rank = curPragma->distribRule.rank;
                    checkDirErrN(rank > 0, 313);
                    checkDirErrN(curPragma->rank < 0 || curPragma->rank == rank, 314);
                    curPragma->rank = rank;
                } else if (tokStr == "align") {
                    checkDirErrN(curPragma->alignFlag == -1, 312);
                    curPragma->alignFlag = 1;
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::l_paren), 315, COLUMN);
                    PP.LexNonComment(Tok);
                    curPragma->alignRule = parseAlignRule(PP, Tok);
                    checkDirErrN(Tok.is(tok::r_paren), 315, COLUMN);
                    PP.LexNonComment(Tok);
                    int rank = curPragma->alignRule.rank;
                    checkDirErrN(rank > 0, 316);
                    checkDirErrN(curPragma->rank < 0 || curPragma->rank == rank, 314);
                    curPragma->rank = rank;
                } else if (tokStr == "shadow") {
                    checkDirErrN(curPragma->shadows.empty(), 317);
                    PP.LexNonComment(Tok);
                    int rank = 0;
                    while (Tok.is(tok::l_square)) {
                        rank++;
                        curPragma->shadows.push_back(readRange(PP, Tok));
                        checkDirErrN(Tok.is(tok::r_square), 318, COLUMN);
                        PP.LexNonComment(Tok);
                    }
                    checkDirErrN(rank > 0, 319);
                    checkDirErrN(curPragma->rank < 0 || curPragma->rank == rank, 314);
                    curPragma->rank = rank;
                } else if (tokStr == "dynamic") {
                    curPragma->dynamicFlag = 1;
                    PP.LexNonComment(Tok);
                } else {
                    checkDirErrN(false, 3110, tokStr.c_str(), COLUMN);
                }
                checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::eod), 305, COLUMN);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.isAnyIdentifier(), 305, COLUMN);
                }
            }
            checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
            checkDirErrN(curPragma->rank > 0, 3111);
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "template") {
        // TEMPLATE
        PragmaTemplate *curPragma = new PragmaTemplate();
        curPragma->copyCommonInfo(this->curPragma);
        curPragma->dynamicFlag = 0;
        PP.LexNonComment(Tok);
        if (!Tok.is(tok::eod)) {
            checkDirErrN(Tok.is(tok::l_square), 3112, COLUMN);
            int rank = 0;
            while (Tok.is(tok::l_square)) {
                rank++;
                MyExpr curSize = readExpr(PP, Tok);
                checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
                curPragma->sizes.push_back(curSize);
                assert(Tok.is(tok::r_square)); // duplicate
                PP.LexNonComment(Tok);
            }
            checkDirErrN(rank > 0, 307, COLUMN);
            curPragma->rank = rank;
            while(Tok.isAnyIdentifier()) {
                tokStr = Tok.getIdentifierInfo()->getName().str();
                if (tokStr == "distribute") {
                    checkDirErrN(curPragma->alignFlag == -1, 312);
                    curPragma->alignFlag = 0;
                    PP.LexNonComment(Tok);
                    curPragma->distribRule = parseDistribRule(PP, Tok);
                    int rank = curPragma->distribRule.rank;
                    checkDirErrN(rank > 0, 313);
                    checkDirErrN(curPragma->rank < 0 || curPragma->rank == rank, 314);
                    curPragma->rank = rank;
                } else if (tokStr == "align") {
                    checkDirErrN(false, 3113);
                    checkDirErrN(curPragma->alignFlag == -1, 312);
                    curPragma->alignFlag = 1;
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::l_paren), 315, COLUMN);
                    PP.LexNonComment(Tok);
                    curPragma->alignRule = parseAlignRule(PP, Tok);
                    checkDirErrN(Tok.is(tok::r_paren), 315, COLUMN);
                    PP.LexNonComment(Tok);
                    int rank = curPragma->alignRule.rank;
                    checkDirErrN(rank > 0, 316);
                    checkDirErrN(curPragma->rank < 0 || curPragma->rank == rank, 314);
                    curPragma->rank = rank;
                } else if (tokStr == "dynamic") {
                    curPragma->dynamicFlag = 1;
                    PP.LexNonComment(Tok);
                } else {
                    checkDirErrN(false, 3114, tokStr.c_str(), COLUMN);
                }
                checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::eod), 305, COLUMN);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.isAnyIdentifier(), 305, COLUMN);
                }
            }
            checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
            checkDirErrN(curPragma->rank > 0, 3111);
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "redistribute") {
        // REDISTRIBUTE
        PragmaRedistribute *curPragma = new PragmaRedistribute();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 324, COLUMN);
        curPragma->name = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        curPragma->distribRule = parseDistribRule(PP, Tok);
        curPragma->rank = curPragma->distribRule.rank;
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        checkDirErrN(curPragma->rank > 0, 325);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "interval") {
        //INTERVAL
        PragmaInterval *curPragma = new PragmaInterval();
        curPragma->copyCommonInfo(this->curPragma);
        curPragma->userID = readExpr(PP, Tok);
        if (curPragma->userID.empty())
            curPragma->userID.append("0");
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "endinterval") {
        //ENDINTERVAL
        PragmaEndInterval *curPragma = new PragmaEndInterval();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "exitinterval") {
        // EXITINTERVAL
        PragmaExitInterval *curPragma = new PragmaExitInterval();
        curPragma->copyCommonInfo(this->curPragma);
        do {
            MyExpr id = readExpr(PP, Tok, ",");
            checkDirErrN(!id.empty(), 391);
            curPragma->ids.push_back(id);
            checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::eod), 3015, COLUMN);
        } while (!Tok.is(tok::eod));
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "realign") {
        // REALIGN
        PragmaRealign *curPragma = new PragmaRealign();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 324, COLUMN);
        curPragma->name = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        curPragma->alignRule = parseAlignRule(PP, Tok);
        curPragma->rank = curPragma->alignRule.rank;
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        if (Tok.isAnyIdentifier()) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            checkDirErrN(tokStr == "new_value", 326, tokStr.c_str(), COLUMN);
            curPragma->newValueFlag = true;
            PP.LexNonComment(Tok);
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        checkDirErrN(curPragma->rank > 0, 325);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "region") {
        // REGION
        PragmaRegion *curPragma = new PragmaRegion();
        curPragma->copyCommonInfo(this->curPragma);
        curPragma->flags = 0;
        curPragma->targets = 0;
        PP.LexNonComment(Tok);
        while (Tok.isAnyIdentifier()) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            if (tokStr == "async") {
                curPragma->flags = curPragma->flags | PragmaRegion::REGION_ASYNC;
                PP.LexNonComment(Tok);
            } else if (tokStr == "targets") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 333, COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    if (tokStr == "HOST")
                        curPragma->targets |= PragmaRegion::DEVICE_TYPE_HOST;
                    else if (tokStr == "CUDA")
                        curPragma->targets |= PragmaRegion::DEVICE_TYPE_CUDA;
                    else
                        checkDirErrN(false, 334);
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 335, COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 336, COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 333, COLUMN);
                PP.LexNonComment(Tok);
            } else {
                int intent = 0;
                std::string clauseName = tokStr;
                if (tokStr == "in")
                    intent = PragmaRegion::INTENT_IN;
                else if (tokStr == "out")
                    intent = PragmaRegion::INTENT_OUT;
                else if (tokStr == "local")
                    intent = PragmaRegion::INTENT_LOCAL;
                else if (tokStr == "inout")
                    intent = PragmaRegion::INTENT_IN | PragmaRegion::INTENT_OUT;
                else if (tokStr == "inlocal")
                    intent = PragmaRegion::INTENT_IN | PragmaRegion::INTENT_LOCAL;
                else
                    checkDirErrN(false, 337, tokStr.c_str(), COLUMN);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 338, clauseName.c_str(), COLUMN);
                PP.LexNonComment(Tok);
                std::vector<SlicedArray> tmpArr = parseSubarrays(PP, Tok);
                for (int i = 0; i < (int)tmpArr.size(); i++)
                    curPragma->regVars.push_back(std::make_pair(tmpArr[i], intent));
                checkDirErrN(Tok.is(tok::r_paren), 338, clauseName.c_str(), COLUMN);
                PP.LexNonComment(Tok);
            }
            checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::eod), 305, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 305, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "parallel") {
        // PARALLEL
        PragmaParallel *curPragma = new PragmaParallel();
        curPragma->copyCommonInfo(this->curPragma);
        curPragma->cudaBlock[0].strExpr = "0";
        curPragma->cudaBlock[1].strExpr = "0";
        curPragma->cudaBlock[2].strExpr = "0";
        curPragma->stage.strExpr = "0";
        PP.LexNonComment(Tok);
        int rank = 0;
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_square) || Tok.isLiteral(), 3410, COLUMN);
        if (Tok.isLiteral()) {
            curPragma->mappedFlag = false;
#if CLANG_VERSION_MAJOR < 4 && CLANG_VERSION_MINOR < 5
            bool ok = sscanf(Tok.getLiteralData(), "%d", &rank) == 1;
            checkDirErrN(ok, 3410, COLUMN);
            PP.LexNonComment(Tok);
#else
            uint64_t val;
            bool ok = PP.parseSimpleIntegerLiteral(Tok, val);
            if (ok)
                rank = val;
            checkDirErrN(ok, 3410, COLUMN);
	    // I am not sure that this check is necessary.
	    // It seems that in the current mode simple Lex() skips comments in pragmas.
	    // So, parseSimpleIntegerLiteral() also skips all comments after literal.
	    if (Tok.is(tok::comment))
              PP.LexNonComment(Tok);
#endif
        } else {
            checkDirErrN(Tok.is(tok::l_square), 307, COLUMN);
            curPragma->mapRule = parseAlignRule(PP, Tok, true);
            curPragma->mappedFlag = curPragma->mapRule.isMapped();
            rank = curPragma->mapRule.rank;
        }
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(rank > 0, 3411);
        curPragma->rank = rank;
        while (Tok.isAnyIdentifier() || Tok.is(tok::kw_private)) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            if (tokStr == "private") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "private", COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    curPragma->privates.push_back(tokStr);
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "private", COLUMN);
                    if (Tok.is(tok::comma))
                        PP.LexNonComment(Tok);
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "private", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "reduction") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "reduction", COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    curPragma->reductions.push_back(ClauseReduction());
                    ClauseReduction *red = &curPragma->reductions.back();
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    red->redType = ClauseReduction::guessRedType(tokStr);
                    checkDirErrN(!red->redType.empty(), 3414, tokStr.c_str(), COLUMN);
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::l_paren), 3412, "reduction", COLUMN);
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.isAnyIdentifier(), 3415, COLUMN);
                    red->arrayName = Tok.getIdentifierInfo()->getName().str();
                    PP.LexNonComment(Tok);
                    if (red->isLoc()) {
                        checkDirErrN(Tok.is(tok::comma), 3416, COLUMN);
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3417, COLUMN);
                        red->locName = Tok.getIdentifierInfo()->getName().str();
                        PP.LexNonComment(Tok);
                        if (Tok.is(tok::comma)) {
                            red->locSize = readExpr(PP, Tok);
                            checkDirErrN(!red->locSize.empty(), 3418);
                            checkDirErrN(isNumber(red->locSize.strExpr), 3419);
                        }
                    }
                    checkDirErrN(Tok.is(tok::r_paren), 3412, "reduction", COLUMN);
                    PP.LexNonComment(Tok);
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "reduction", COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3420, COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "reduction", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "shadow_renew") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "shadow_renew", COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    curPragma->shadowRenews.push_back(ClauseShadowRenew());
                    ClauseShadowRenew *shad = &curPragma->shadowRenews.back();
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    shad->cornerFlag = 0;
                    shad->arrayName = tokStr;
                    PP.LexNonComment(Tok);
                    if (Tok.is(tok::l_square)) {
                        // Shadow widths are present
                        int rank = 0;
                        while (Tok.is(tok::l_square)) {
                            rank++;
                            shad->shadows.push_back(AxisShadow());
                            AxisShadow &shadow = shad->shadows.back();
                            PP.EnableBacktrackAtThisPos();
                            PP.LexNonComment(Tok);
                            if (Tok.is(tok::string_literal)) {
                                // Indirect shadow case
                                PP.CommitBacktrackedTokens();
                                shadow.isIndirect = true;
                                shad->isIndirect = true;
                                while (Tok.is(tok::string_literal)) {
                                    std::string name;
                                    PP.FinishLexStringLiteral(Tok, name, "", true);
                                    shadow.names.push_back(name);
                                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_square), 3015, COLUMN);
                                    if (Tok.is(tok::comma)) {
                                        PP.LexNonComment(Tok);
                                        checkDirErrN(Tok.is(tok::string_literal), 3015, COLUMN);
                                    }
                                }
                            } else {
                                // Width case
                                PP.Backtrack();
                                shadow.isIndirect = false;
                                std::pair<MyExpr, MyExpr> widhts = readRange(PP, Tok);
                                shadow.lower = widhts.first;
                                shadow.upper = widhts.second;
                            }
                            checkDirErrN(Tok.is(tok::r_square), 3421, "shadow_renew", COLUMN); // Never happens
                            PP.LexNonComment(Tok);
                        }
                        assert(rank > 0); // is consequent of upper part
                        shad->rank = rank;
                    } else {
                        // Shadow widths are absent
                        shad->rank = -1;
                    }
                    if (Tok.is(tok::l_paren)) {
                        // CORNER
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3422, COLUMN);
                        checkDirErrN(Tok.getIdentifierInfo()->getName() == "corner", 3422, COLUMN);
                        checkDirErr(!shad->isIndirect, "Corner specification is not allowed when renewing indirect shadow edges");
                        shad->cornerFlag = 1;
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.is(tok::r_paren), 3412, "shadow_renew", COLUMN);
                        PP.LexNonComment(Tok);
                    }
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "shadow_renew", COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3423, "shadow_renew", COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "shadow_renew", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "across") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "across", COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    curPragma->acrosses.push_back(ClauseAcross());
                    ClauseAcross *acr = &curPragma->acrosses.back();
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    if (tokStr == "out") {
                        PP.EnableBacktrackAtThisPos();
                        PP.LexNonComment(Tok);
                        if (Tok.is(tok::colon)) {
                            acr->isOut = true;
                            PP.CommitBacktrackedTokens();
                            PP.LexNonComment(Tok);
                            checkDirErrN(Tok.isAnyIdentifier(), 3423, "across", COLUMN);
                            tokStr = Tok.getIdentifierInfo()->getName().str();
                        } else {
                            PP.Backtrack();
                        }
                    }
                    acr->arrayName = tokStr;
                    PP.LexNonComment(Tok);
                    if (Tok.is(tok::l_square)) {
                        // Lengths are present
                        int rank = 0;
                        while (Tok.is(tok::l_square)) {
                            rank++;
                            acr->widths.push_back(readRange(PP, Tok));
                            checkDirErrN(Tok.is(tok::r_square), 3421, "across", COLUMN); // Never happens
                            PP.LexNonComment(Tok);
                        }
                        assert(rank > 0); // is consequent of upper part
                        acr->rank = rank;
                    } else {
                        // Lengths are absent
                        acr->rank = -1;
                    }
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "across", COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3423, "across", COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "across", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "remote_access") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "remote_access", COLUMN);
                while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::eod)) {
                    curPragma->rmas.push_back(parseOneRma(PP, Tok, curPragma->mapRule.nameToAxis));
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "remote_access", COLUMN);
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "remote_access", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "cuda_block") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "cuda_block", COLUMN);
                curPragma->cudaBlock[0].strExpr = "1";
                curPragma->cudaBlock[1].strExpr = "1";
                curPragma->cudaBlock[2].strExpr = "1";
                for (int i = 0; i < 3; i++) {
                    MyExpr val = readExpr(PP, Tok, ",");
                    checkDirErrN(!val.empty(), 3425, i + 1);
                    checkDirErrN(Tok.is(tok::r_paren) || Tok.is(tok::comma), 3413, "cuda_block", COLUMN);
                    curPragma->cudaBlock[i] = val;
                    if (Tok.is(tok::r_paren))
                        break;
                    else
                        checkDirErrN(Tok.is(tok::comma), 3413, "cuda_block", COLUMN);
                }
                checkDirErrN(Tok.isNot(tok::comma), 3426);
                checkDirErrN(Tok.is(tok::r_paren), 3412, "cuda_block", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "stage") {
                checkDirErrN(curPragma->mappedFlag, 3427);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "stage", COLUMN);
                curPragma->stage = readExpr(PP, Tok);
                checkDirErrN(!curPragma->stage.empty(), 3428);
                checkDirErrN(Tok.is(tok::r_paren), 3412, "stage", COLUMN);
                PP.LexNonComment(Tok);
            } else if (tokStr == "tie") {
                checkDirErrN(curPragma->mapRule.isInitialized(), 3441);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 3412, "tie", COLUMN);
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    curPragma->ties.push_back(ClauseTie());
                    ClauseTie *tie = &curPragma->ties.back();
                    tokStr = Tok.getIdentifierInfo()->getName().str();
                    tie->arrayName = tokStr;
                    PP.LexNonComment(Tok);
                    while (Tok.is(tok::l_square)) {
                        PP.LexNonComment(Tok);
                        if (Tok.is(tok::r_square)) {
                            tie->loopAxes.push_back(0);
                        } else if (Tok.is(tok::minus)) {
                            PP.LexNonComment(Tok);
                            checkDirErrN(Tok.isAnyIdentifier(), 3442, "tie", COLUMN);
                            tokStr = Tok.getIdentifierInfo()->getName().str();
                            checkDirErrN(curPragma->mapRule.nameToAxis.find(tokStr) != curPragma->mapRule.nameToAxis.end(), 3442, "tie", COLUMN);
                            tie->loopAxes.push_back(-curPragma->mapRule.nameToAxis[tokStr]);
                            PP.LexNonComment(Tok);
                        } else {
                            checkDirErrN(Tok.isAnyIdentifier(), 3442, "tie", COLUMN);
                            tokStr = Tok.getIdentifierInfo()->getName().str();
                            checkDirErrN(curPragma->mapRule.nameToAxis.find(tokStr) != curPragma->mapRule.nameToAxis.end(), 3442, "tie", COLUMN);
                            tie->loopAxes.push_back(curPragma->mapRule.nameToAxis[tokStr]);
                            PP.LexNonComment(Tok);
                        }
                        checkDirErrN(Tok.is(tok::r_square), 3421, "tie", COLUMN);
                        PP.LexNonComment(Tok);
                    }
                    checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3413, "tie", COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErrN(Tok.isAnyIdentifier(), 3423, "tie", COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 3412, "tie", COLUMN);
                PP.LexNonComment(Tok);
            } else {
                checkDirErrN(false, 3429, tokStr.c_str(), COLUMN);
            }
            checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::kw_private) || Tok.is(tok::eod), 305, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier() || Tok.is(tok::kw_private), 305, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "get_actual" || tokStr == "actual") {
        // GET_ACTUAL or ACTUAL
        PragmaGetSetActual *curPragma = new PragmaGetSetActual(tokStr == "get_actual" ? DvmPragma::pkGetActual : DvmPragma::pkSetActual);
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        curPragma->vars = parseSubarrays(PP, Tok);
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "inherit") {
        // INHERIT
        PragmaInherit *curPragma = new PragmaInherit();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        while (Tok.isAnyIdentifier()) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            curPragma->names.push_back(tokStr);
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 363, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 363, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "remote_access") {
        // REMOTE_ACCESS
        PragmaRemoteAccess *curPragma = new PragmaRemoteAccess();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        std::map<std::string, int> noAxesNames; // Empty map, because there are no axes on which can be mapped dimension of array
        while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::eod)) {
            curPragma->rmas.push_back(parseOneRma(PP, Tok, noAxesNames));
            checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 372, COLUMN);
        }
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "host_section") {
        // HOST_SECTION
        PragmaHostSection *curPragma = new PragmaHostSection();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "instantiations") {
        // C++ function template annotation
        PragmaInstantiations *curPragma = new PragmaInstantiations();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        while (Tok.isAnyIdentifier()) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            checkDirErrN(tokStr == "set", 364, tokStr.c_str(), COLUMN);
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
            std::map<std::string, std::string> newSet;
            PP.LexNonComment(Tok);
            while (Tok.isAnyIdentifier()) {
                std::string paramName = Tok.getIdentifierInfo()->getName().str();
                checkDirErrN(newSet.find(paramName) == newSet.end(), 366, paramName.c_str(), COLUMN);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::equal), 365, COLUMN);
                PP.LexNonComment(Tok);
                std::string paramValue;
                while (Tok.isNot(tok::comma) && Tok.isNot(tok::r_paren) && Tok.isNot(tok::eod)) {
                    paramValue += (paramValue.empty() ? "" : " ") + PP.getSpelling(Tok);
                    PP.LexNonComment(Tok);
                }
                newSet[paramName] = paramValue;
                checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 367, COLUMN);
                if (Tok.is(tok::comma)) {
                    PP.LexNonComment(Tok);
                }
                checkDirErrN(Tok.isAnyIdentifier() || Tok.is(tok::r_paren), 367, COLUMN);
            }
            checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
            PP.LexNonComment(Tok);
            curPragma->valueSets.insert(newSet);
            checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::eod), 305, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 305, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "shadow_add") {
        // SHADOW_ADD
        PragmaShadowAdd *curPragma = new PragmaShadowAdd();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        curPragma->targetName = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        int rank = 0;
        curPragma->ruleAxis = 0; // 1-based
        while (Tok.is(tok::l_square)) {
            rank++;
            PP.EnableBacktrackAtThisPos();
            Token tmpTok;
            PP.LexNonComment(tmpTok);
            PP.Backtrack();
            if (!tmpTok.is(tok::r_square)) {
                checkDirErr(curPragma->ruleAxis <= 0, "Only one axis can have shadow specification");
                curPragma->ruleAxis = rank;
                curPragma->rule = parseDerivedAxisRule(PP, Tok);
            } else {
                PP.LexNonComment(Tok);
            }
            checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
            PP.LexNonComment(Tok);
        }
        checkDirErr(rank > 0, "Rank must be positive");
        checkDirErr(curPragma->ruleAxis > 0, "One axis must specify a shadow edge");
        curPragma->rank = rank;
        checkDirErr(Tok.is(tok::equal), "Equal sign is expected at column %d", COLUMN);
        PP.LexNonComment(Tok);
        checkDirErr(Tok.is(tok::string_literal), "Name of the shadow edge is expected at column %d", COLUMN);
        PP.FinishLexStringLiteral(Tok, curPragma->shadowName, "", true);
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        while (Tok.isAnyIdentifier()) {
            // clauses
            tokStr = Tok.getIdentifierInfo()->getName().str();
            if (tokStr == "include_to") {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
                // Variable name list
                PP.LexNonComment(Tok);
                while (Tok.isAnyIdentifier()) {
                    curPragma->includeList.push_back(Tok.getIdentifierInfo()->getName().str());
                    PP.LexNonComment(Tok);
                    checkDirErr(Tok.is(tok::comma) || Tok.is(tok::r_paren), "Comma-separated list of variables expected at column %d", COLUMN);
                    if (Tok.is(tok::comma)) {
                        PP.LexNonComment(Tok);
                        checkDirErr(Tok.isAnyIdentifier(), "Comma-separated list of variables expected at column %d", COLUMN);
                    }
                }
                checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
                PP.LexNonComment(Tok);
            } else {
                checkDirErr(false, "Unknown shadow_add clause '%s' at column %d", tokStr.c_str(), COLUMN);
            }
            checkDirErrN(Tok.is(tok::comma) || Tok.isAnyIdentifier() || Tok.is(tok::eod), 305, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 305, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "localize") {
        // LOCALIZE
        PragmaLocalize *curPragma = new PragmaLocalize();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        curPragma->refName = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::equal), 3012, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::greater), 3012, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        curPragma->targetName = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        int rank = 0;
        while (Tok.is(tok::l_square)) {
            rank++;
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.is(tok::colon) || Tok.is(tok::r_square), 381, COLUMN);
            if (Tok.is(tok::colon)) {
                checkDirErrN(curPragma->targetAxis <= 0, 382);
                curPragma->targetAxis = rank;
                PP.LexNonComment(Tok);
            }
            checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
            PP.LexNonComment(Tok);
        }
        curPragma->targetRank = rank;
        checkDirErrN(rank == 0 || curPragma->targetAxis > 0, 382);
        checkDirErrN(Tok.is(tok::r_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "unlocalize") {
        // UNLOCALIZE
        PragmaUnlocalize *curPragma = new PragmaUnlocalize();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        while (Tok.isAnyIdentifier()) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            curPragma->nameList.push_back(tokStr);
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3015, COLUMN);
            if (Tok.is(tok::comma)) {
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 3015, COLUMN);
            }
        }
        checkDirErrN(Tok.is(tok::r_paren), 3015, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else if (tokStr == "array_copy") {
        // ARRAY_COPY
        // XXX: Experimental temporary directive
        // Syntax: array_copy(ARR1 = ARR2)
        PragmaArrayCopy *curPragma = new PragmaArrayCopy();
        curPragma->copyCommonInfo(this->curPragma);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_paren), 308, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        tokStr = Tok.getIdentifierInfo()->getName().str();
        curPragma->dstName = tokStr;
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::equal), 3016, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        tokStr = Tok.getIdentifierInfo()->getName().str();
        curPragma->srcName = tokStr;
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::r_paren), 3015, COLUMN);
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::eod), 306, COLUMN);
        fileCtx.addPragma(fileID.getHashValue(), curPragma);
    } else {
        checkDirErrN(false, 309, tokStr.c_str());
    }
    delete curPragma;
}

DerivedAxisRule DvmPragmaHandler::parseDerivedAxisRule(PreprocessorImpl &PP, Token &Tok) {
    // At entrance: First unlexed token is the part of the rule.
    // At exit: Lexed one token after the end of the rule. Probably parenthesis, comma, or smth else. Can't be l_square.
    // TODO: All the messages mention distribution rule. Can be confusing if used to parse a shadow edge.
    // TODO: Introduce specialized message codes in the group 32*
    DerivedAxisRule res;
    std::set<std::string> externalNames;
    for (;;) {
        MyExpr e = readExpr(PP, Tok, ", with");
        res.exprs.push_back(e);
        externalNames.insert(e.usedNames.begin(), e.usedNames.end());
        if (Tok.isNot(tok::comma))
            break;
    }
    std::string tokStr = PP.getSpelling(Tok);
    checkDirErrN(tokStr == "with", 3224, COLUMN);
    PP.LexNonComment(Tok);
    checkDirErrN(Tok.isAnyIdentifier(), 3225, COLUMN);
    res.templ = PP.getSpelling(Tok);
    PP.LexNonComment(Tok);
    int rank = 0;
    std::set<std::string> seenDummies;
    while (Tok.is(tok::l_square)) {
        rank++;
        PP.EnableBacktrackAtThisPos();
        PP.LexNonComment(Tok);
        DerivedRHSExpr rhsExpr;
        if (PP.getSpelling(Tok) == "@") {
            PP.CommitBacktrackedTokens();
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.isAnyIdentifier(), 3221, COLUMN);
            std::string dummy = PP.getSpelling(Tok);
            checkDirErrN(seenDummies.insert(dummy).second, 3222, dummy.c_str(), COLUMN);
            externalNames.erase(dummy);
            rhsExpr.dummyName = dummy;
            PP.LexNonComment(Tok);
            while (Tok.is(tok::plus)) {
                PP.LexNonComment(Tok);
                checkDirErr(Tok.is(tok::string_literal), "String literal expected at column %d", COLUMN);
                std::string name;
                PP.FinishLexStringLiteral(Tok, name, "", true);
                rhsExpr.addShadows.push_back(name);
                checkDirErr(Tok.is(tok::plus) || Tok.is(tok::r_square), "Plus expected at column %d", COLUMN);
            }
        } else {
            PP.Backtrack();
            rhsExpr.constExpr = readExpr(PP, Tok);
        }
        res.rhsExprs.push_back(rhsExpr);
        checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
        PP.LexNonComment(Tok);
    }
    assert(rank == (int)res.rhsExprs.size());
    res.externalNames = externalNames;
    return res;
}

DistribRule DvmPragmaHandler::parseDistribRule(PreprocessorImpl &PP, Token &Tok) {
    DistribRule res;
    std::string tokStr;
    while (Tok.is(tok::l_square)) {
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.isAnyIdentifier() || Tok.is(tok::star) || Tok.is(tok::r_square), 327, COLUMN);
        DistribAxisRule axisRule;
        if (Tok.is(tok::star) || Tok.is(tok::r_square)) {
            axisRule.distrType = DistribAxisRule::dtReplicated;
            if (Tok.is(tok::star))
                PP.LexNonComment(Tok);
        } else {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            if (tokStr == "block")
                axisRule.distrType = DistribAxisRule::dtBlock;
            else if (tokStr == "wgtblock") {
                axisRule.distrType = DistribAxisRule::dtWgtBlock;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 328, "wgtblock", COLUMN);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 329, "wgtblock", COLUMN);
                std::string wgtArrName = Tok.getIdentifierInfo()->getName().str();
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::comma), 3210, COLUMN);
                MyExpr wgtArrSize = readExpr(PP, Tok);
                checkDirErrN(!wgtArrSize.empty(), 3211, "wgtblock");
                checkDirErrN(Tok.is(tok::r_paren), 328, "wgtblock", COLUMN);
                axisRule.wgtBlockArray = std::make_pair(wgtArrName, wgtArrSize);
            } else if (tokStr == "genblock") {
                axisRule.distrType = DistribAxisRule::dtGenBlock;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 328, "genblock", COLUMN);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 329, "genblock", COLUMN);
                tokStr = Tok.getIdentifierInfo()->getName().str();
                axisRule.genBlockArray = tokStr;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::r_paren), 328, "genblock", COLUMN);
            } else if (tokStr == "multblock") {
                axisRule.distrType = DistribAxisRule::dtMultBlock;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 328, "multblock", COLUMN);
                axisRule.multBlockValue = readExpr(PP, Tok);
                checkDirErrN(!axisRule.multBlockValue.empty(), 3211, "multblock");
                checkDirErrN(Tok.is(tok::r_paren), 328, "multblock", COLUMN);
            } else if (tokStr == "indirect") {
                axisRule.distrType = DistribAxisRule::dtIndirect;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 328, "indirect", COLUMN);
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.isAnyIdentifier(), 329, "indirect", COLUMN);
                tokStr = Tok.getIdentifierInfo()->getName().str();
                axisRule.indirectArray = tokStr;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::r_paren), 328, "indirect", COLUMN);
            } else if (tokStr == "derived") {
                axisRule.distrType = DistribAxisRule::dtDerived;
                PP.LexNonComment(Tok);
                checkDirErrN(Tok.is(tok::l_paren), 328, "derived", COLUMN);
                axisRule.derivedRule = parseDerivedAxisRule(PP, Tok);
                checkDirErrN(Tok.is(tok::r_paren), 328, "derived", COLUMN);
            } else {
                checkDirErrN(false, 3212, tokStr.c_str(), COLUMN);
            }
            PP.LexNonComment(Tok);
        }
        res.axes.push_back(axisRule);
        checkDirErrN(Tok.is(tok::r_square), 3213, COLUMN);
        PP.LexNonComment(Tok);
    }
    res.rank = res.axes.size();
    return res;
}

AlignAxisRule DvmPragmaHandler::parseAlignAxisRule(PreprocessorImpl &PP, Token &Tok, std::map<std::string, int> nameToAxis, bool parLoopFlag) {
    AlignAxisRule res;
    AlignAxisRule *rule = &res;
    PP.EnableBacktrackAtThisPos();
    rule->origExpr = readExpr(PP, Tok);
    PP.Backtrack();
    PP.LexNonComment(Tok);
    std::vector<MyExpr> s;
    int axesSeen = 0;
    while (Tok.isNot(tok::r_square) && Tok.isNot(tok::eod)) {
        if (Tok.is(tok::l_paren))
            s.push_back(readExpr(PP, Tok));
        else {
            MyExpr simpleExpr;
            simpleExpr.strExpr = PP.getSpelling(Tok);
            if (Tok.isAnyIdentifier())
                simpleExpr.usedNames.insert(simpleExpr.strExpr);
            s.push_back(simpleExpr);
            if (nameToAxis.find(s.back().strExpr) != nameToAxis.end())
                axesSeen++;
        }
        PP.LexNonComment(Tok);
    }
    int partCount = s.size();
    if (parLoopFlag)
        checkDirErrN(axesSeen <= 1, 3430);
    else
        checkDirErrN(axesSeen <= 1, 3215);
    if (axesSeen == 0) {
        // Replicated or to constant
        if (partCount == 0)
            rule->axisNumber = -1;
        else {
            rule->axisNumber = 0;
            rule->multiplier.strExpr = "0";
            rule->summand = rule->origExpr;
        }
    } else {
        // Linear rule
        bool ok = partCount == 1 || partCount == 3 || partCount == 5;
        if (parLoopFlag)
            checkDirErrN(ok, 3431, COLUMN);
        else
            checkDirErrN(ok, 3216, COLUMN);
        if (partCount == 1) {
            assert(nameToAxis.find(s[0].strExpr) != nameToAxis.end()); // is consequent of upper part
            rule->axisNumber = nameToAxis[s[0].strExpr];
            rule->multiplier.strExpr = "1";
            rule->summand.strExpr = "0";
        } else if (partCount == 3) {
            if (s[1].strExpr == "*") {
                ok = nameToAxis.find(s[2].strExpr) != nameToAxis.end();
                if (parLoopFlag)
                    checkDirErrN(ok, 3432, COLUMN);
                else
                    checkDirErrN(ok, 3217, COLUMN);
                rule->axisNumber = nameToAxis[s[2].strExpr];
                rule->multiplier = s[0];
                rule->summand.strExpr = "0";
            } else if (s[1].strExpr == "+" || s[1].strExpr == "-") {
                ok = nameToAxis.find(s[0].strExpr) != nameToAxis.end();
                if (parLoopFlag)
                    checkDirErrN(ok, 3433, COLUMN);
                else
                    checkDirErrN(ok, 3218, COLUMN);
                rule->axisNumber = nameToAxis[s[0].strExpr];
                rule->multiplier.strExpr = "1";
                rule->summand = s[2];
                if (s[1].strExpr == "-") {
                    rule->summand.prepend("-(");
                    rule->summand.append(")");
                }
            } else {
                if (parLoopFlag)
                    checkDirErrN(false, 3431, COLUMN);
                else
                    checkDirErrN(false, 3216, COLUMN);
            }
        } else {
            ok = s[1].strExpr == "*" && (s[3].strExpr == "+" || s[3].strExpr == "-");
            if (parLoopFlag)
                checkDirErrN(ok, 3431, COLUMN);
            else
                checkDirErrN(ok, 3216, COLUMN);
            ok = nameToAxis.find(s[2].strExpr) != nameToAxis.end();
            if (parLoopFlag)
                checkDirErrN(ok, 3434, COLUMN);
            else
                checkDirErrN(ok, 3219, COLUMN);
            rule->axisNumber = nameToAxis[s[2].strExpr];
            rule->multiplier = s[0];
            rule->summand = s[4];
            if (s[3].strExpr == "-") {
                rule->summand.prepend("-(");
                rule->summand.append(")");
            }
        }
    }
    if (parLoopFlag)
        checkDirErrN(Tok.is(tok::r_square), 3435, COLUMN);
    else
        checkDirErrN(Tok.is(tok::r_square), 3220, COLUMN);
    return res;
}

AlignRule DvmPragmaHandler::parseAlignRule(PreprocessorImpl &PP, Token &Tok, bool parLoopFlag) {
    AlignRule res;
    std::string tokStr;
    int rank = 0;
    while (Tok.is(tok::l_square)) {
        rank++;
        PP.LexNonComment(Tok);
        if (parLoopFlag)
            checkDirErrN(Tok.isAnyIdentifier(), 3436, COLUMN);
        else
            checkDirErrN(Tok.isAnyIdentifier() || Tok.is(tok::r_square), 3221, COLUMN);
        if (!Tok.is(tok::r_square)) {
            tokStr = Tok.getIdentifierInfo()->getName().str();
            bool ok = res.nameToAxis.find(tokStr) == res.nameToAxis.end();
            if (parLoopFlag)
                checkDirErrN(ok, 3437, tokStr.c_str(), COLUMN);
            else
                checkDirErrN(ok, 3222, tokStr.c_str(), COLUMN);
            res.nameToAxis[tokStr] = rank;
            PP.LexNonComment(Tok);
        }
        if (parLoopFlag)
            checkDirErrN(Tok.is(tok::r_square), 3438, COLUMN);
        else
            checkDirErrN(Tok.is(tok::r_square), 3223, COLUMN);
        PP.LexNonComment(Tok);
    }
    res.rank = rank;
    if (parLoopFlag) {
        if (Tok.is(tok::r_paren)) {
            return res;
        }
        checkDirErrN(Tok.isAnyIdentifier(), 3439, COLUMN);
    } else {
        checkDirErrN(Tok.isAnyIdentifier(), 3224, COLUMN);
    }
    tokStr = Tok.getIdentifierInfo()->getName().str();
    if (parLoopFlag)
        checkDirErrN(tokStr == "on", 3439, COLUMN);
    else
        checkDirErrN(tokStr == "with", 3224, COLUMN);
    PP.LexNonComment(Tok);
    if (parLoopFlag)
        checkDirErrN(Tok.isAnyIdentifier(), 3440, COLUMN);
    else
        checkDirErrN(Tok.isAnyIdentifier(), 3225, COLUMN);
    res.templ = Tok.getIdentifierInfo()->getName().str();
    int templRank = 0;
    PP.LexNonComment(Tok);
    while (Tok.is(tok::l_square)) {
        templRank++;
        res.axisRules.push_back(parseAlignAxisRule(PP, Tok, res.nameToAxis, parLoopFlag));
        if (parLoopFlag)
            checkDirErrN(Tok.is(tok::r_square), 3435, COLUMN);
        else
            checkDirErrN(Tok.is(tok::r_square), 3220, COLUMN);
        PP.LexNonComment(Tok);
    }
    res.templRank = templRank;
    return res;
}

std::vector<SlicedArray> DvmPragmaHandler::parseSubarrays(PreprocessorImpl &PP, Token &Tok) {
    std::vector<SlicedArray> res;
    while (Tok.isAnyIdentifier()) {
        res.push_back(SlicedArray());
        SlicedArray *curVar = &res.back();
        curVar->name = Tok.getIdentifierInfo()->getName().str();
        PP.LexNonComment(Tok);
        checkDirErrN(Tok.is(tok::l_square) || Tok.is(tok::comma) || Tok.is(tok::r_paren), 3012, COLUMN);
        if (Tok.is(tok::l_square)) {
            // Subscript is present
            while (Tok.is(tok::l_square)) {
                curVar->bounds.push_back(readRange(PP, Tok));
                checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
                PP.LexNonComment(Tok);
            }
        } else {
            // Subscript is absent
            curVar->slicedFlag = 0;
        }
        checkDirErrN(Tok.is(tok::comma) || Tok.is(tok::r_paren), 3013, COLUMN);
        if (Tok.is(tok::comma)) {
            PP.LexNonComment(Tok);
            checkDirErrN(Tok.isAnyIdentifier(), 3014, COLUMN);
        }
    }
    return res;
}

MyExpr DvmPragmaHandler::readExpr(PreprocessorImpl &PP, Token &Tok, std::string stopTokens) {
    MyExpr res;
    if (!stopTokens.empty()) {
        if (stopTokens[0] != ' ')
            stopTokens = " " + stopTokens;
        if (*stopTokens.rbegin() != ' ')
            stopTokens += " ";
    }
    bool hasDelim = true;
    int depth = 1;
    bool hasDot = false;
    std::vector<std::pair<int, int> > activeArrayRefs; // first = depth, second = array reference index in res.arrayRefs
    std::vector<RangeDesc *> activeRanges;
    std::vector<int> activeQuestionMarks;
    RangeDesc *range = new RangeDesc;
    range->beginValue.first = 0;
    range->endValue.first = -1;
    res.ranges.push_back(range);
    activeRanges.push_back(range);
    activeQuestionMarks.push_back(0);
    int braceDepth = 0;
    for (;;) {
        PP.LexNonComment(Tok);
        std::string tokStr = PP.getSpelling(Tok);
        if (Tok.is(tok::l_square) || Tok.is(tok::l_paren)) {
            depth++;
            activeQuestionMarks.push_back(0);
        } else if (Tok.is(tok::r_square) || Tok.is(tok::r_paren)) {
            depth--;
            activeQuestionMarks.pop_back();
            if (braceDepth == 0) {
                RangeDesc *range = activeRanges.back();
                if (range->endValue.first >= 0)
                    range->endValue.second = res.strExpr.size();
                activeRanges.pop_back();
            }
        }
        if (Tok.is(tok::l_brace))
            braceDepth++;
        else if (Tok.is(tok::r_brace))
            braceDepth--;
        if (depth == 0)
            break;
        if (depth == 1 && strstr(stopTokens.c_str(), (" " + tokStr + " ").c_str()))
            break;
        if (Tok.is(tok::eod)) {
            checkDirErrN(depth == 1, 3010, COLUMN);
            break;
        }
        if (Tok.is(tok::question))
            activeQuestionMarks.back()++;
        if (Tok.is(tok::colon) && activeQuestionMarks.back() == 0 && braceDepth == 0)
            activeRanges.back()->beginValue.second = (int)res.strExpr.size() - 1;
        bool noNeed = Tok.is(tok::period) || Tok.is(tok::periodstar) || Tok.is(tok::l_brace) || Tok.is(tok::l_paren) || Tok.is(tok::l_square) ||
                Tok.is(tok::r_brace) || Tok.is(tok::r_paren) || Tok.is(tok::r_square) || Tok.is(tok::arrow) || Tok.is(tok::arrowstar) || Tok.is(tok::tilde)
                || Tok.is(tok::exclaim) || Tok.is(tok::plusplus) || Tok.is(tok::minusminus) || Tok.isAnyIdentifier() || Tok.isLiteral() ||
                Tok.is(tok::kw_sizeof) || Tok.is(tok::kw___func__);
        bool needLeft = !(noNeed || Tok.is(tok::comma));
        bool needRight = !(noNeed);
        if (needLeft && !hasDelim)
            res.strExpr += " ";
        int curIdx = res.strExpr.size();
        if (Tok.is(tok::l_square) && !activeArrayRefs.empty() && activeArrayRefs.back().first == depth - 1) {
            res.arrayRefs[activeArrayRefs.back().second].indexes.push_back(std::make_pair(curIdx, curIdx));
        }
        if (Tok.is(tok::r_square) && !activeArrayRefs.empty() && activeArrayRefs.back().first == depth) {
            res.arrayRefs[activeArrayRefs.back().second].indexes.back().second = curIdx;
            Token tmpTok;
            PP.EnableBacktrackAtThisPos();
            PP.LexNonComment(tmpTok);
            PP.Backtrack();
            // LookAhead(0) is not applicable because we ignore comments
            if (tmpTok.isNot(tok::l_square))
                activeArrayRefs.pop_back();
        }
        if (Tok.isAnyIdentifier() && !hasDot) {
            std::string tokStr = Tok.getIdentifierInfo()->getName().str();
            res.usedNames.insert(tokStr);
            if (activeArrayRefs.empty())
                res.topLevelNames.insert(tokStr);
            Token tmpTok;
            PP.EnableBacktrackAtThisPos();
            PP.LexNonComment(tmpTok);
            PP.Backtrack();
            // LookAhead(0) is not applicable because we ignore comments
            if (tmpTok.is(tok::l_square)) {
                ArrayRefDesc newRef;
                newRef.name = tokStr;
                newRef.head = std::make_pair(curIdx, curIdx + newRef.name.size() - 1);
                res.arrayRefs.push_back(newRef);
                activeArrayRefs.push_back(std::make_pair(depth, res.arrayRefs.size() - 1));
            }
        }
        res.strExpr += tokStr;
        if (needRight) {
            res.strExpr += " ";
            hasDelim = true;
        } else
            hasDelim = false;
        if (Tok.is(tok::colon) && activeQuestionMarks.back() == 0 && braceDepth == 0)
            activeRanges.back()->endValue.first = (int)res.strExpr.size();
        if (Tok.is(tok::colon) && activeQuestionMarks.back() > 0)
            activeQuestionMarks.back()--;
        if (braceDepth == 0 && (Tok.is(tok::l_paren) || Tok.is(tok::l_square))) {
            RangeDesc *range = new RangeDesc;
            range->beginValue.first = res.strExpr.size();
            range->endValue.first = -1;
            activeRanges.back()->children.push_back(range);
            activeRanges.push_back(range);
        }
        hasDot = Tok.is(tok::period) || Tok.is(tok::periodstar) || Tok.is(tok::arrow) || Tok.is(tok::arrowstar);
    }
    if (!res.strExpr.empty() && hasDelim)
        res.strExpr.resize(res.strExpr.size() - 1);
    if (!res.ranges[0]->collapse()) {
        RangeDesc *range = res.ranges[0];
        res.ranges = range->children;
        range->children.clear();
        delete range;
    }
    return res;
}

std::pair<MyExpr, MyExpr> DvmPragmaHandler::readRange(PreprocessorImpl &PP, Token &Tok) {
    MyExpr lb, hb;
    lb = readExpr(PP, Tok, ":");
    checkDirErrN(Tok.is(tok::r_square) || Tok.is(tok::colon), 3011, COLUMN);
    if (Tok.is(tok::colon)) {
        hb = readExpr(PP, Tok);
        checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
    } else {
        // TODO: be aware of side-effects, maybe
        hb = lb;
    }
    return std::make_pair(lb, hb);
}

ClauseRemoteAccess DvmPragmaHandler::parseOneRma(PreprocessorImpl &PP, Token &Tok, const std::map<std::string, int> &nameToAxis) {
    ClauseRemoteAccess res;
    PP.LexNonComment(Tok);
    checkDirErrN(Tok.isAnyIdentifier(), 373, COLUMN);
    res.arrayName = PP.getSpelling(Tok);
    PP.LexNonComment(Tok);
    int rank = 0;
    int nonConstRank = 0;
    while (Tok.is(tok::l_square)) {
        rank++;
        res.axisRules.push_back(parseAlignAxisRule(PP, Tok, nameToAxis, true));
        if (res.axisRules.back().axisNumber != 0) {
            nonConstRank++;
            res.axes.push_back(rank);
        }
        checkDirErrN(Tok.is(tok::r_square), 307, COLUMN);
        PP.LexNonComment(Tok);
    }
    checkDirErrN(rank > 0, 374, COLUMN);
    res.rank = rank;
    res.nonConstRank = nonConstRank;
    return res;
}

}
