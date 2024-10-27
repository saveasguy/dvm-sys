#include "external_preprocessor.h"
#include <clang/Basic/Version.h>
#include <clang/Lex/LiteralSupport.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

using namespace clang;

namespace cdvmh {
bool ExternalPreprocessor::parseSimpleIntegerLiteral(
    clang::Token &Tok, uint64_t &Value) {
  assert(Tok.is(tok::numeric_constant));
  SmallString<8> IntegerBuffer;
  bool NumberInvalid = false;
  StringRef Spelling = getSpelling(Tok, IntegerBuffer, &NumberInvalid);
  if (NumberInvalid)
    return false;
#if CLANG_VERSION_MAJOR > 10
  NumericLiteralParser Literal(Spelling, Tok.getLocation(), getSourceManager(),
                               getLangOpts(), getTargetInfo(),
                               getDiagnostics());
#else
  NumericLiteralParser Literal(Spelling, Tok.getLocation(), *mPP);
#endif
  if (Literal.hadError || !Literal.isIntegerLiteral() || Literal.hasUDSuffix())
    return false;
  llvm::APInt APVal(64, 0);
  if (Literal.GetIntegerValue(APVal))
    return false;
  Lex(Tok);
  Value = APVal.getLimitedValue();
  return true;
}

bool ExternalPreprocessor::FinishLexStringLiteral(Token &Result,
    std::string &String, const char *DiagnosticTag, bool AllowMacroExpansion) {
  // We need at least one string literal.
  if (Result.isNot(tok::string_literal)) {
    Diag(Result, diag::err_expected_string_literal)
      << /*Source='in...'*/0 << DiagnosticTag;
    return false;
  }

  // Lex string literal tokens, optionally with macro expansion.
  SmallVector<Token, 4> StrToks;
  do {
    StrToks.push_back(Result);

    if (Result.hasUDSuffix())
      Diag(Result, diag::err_invalid_string_udl);

    if (AllowMacroExpansion)
      Lex(Result);
    else
      LexUnexpandedToken(Result);
  } while (Result.is(tok::string_literal));

  // Concatenate and parse the strings.
  // Concatenate and parse the strings.
#if LLVM_VERSION_MAJOR < 18
  StringLiteralParser Literal(StrToks, *mPP);
#else  
  llvm::ArrayRef<Token> StrToksArray(StrToks);
  StringLiteralParser Literal(StrToksArray, *mPP);
#endif

#if LLVM_VERSION_MAJOR < 15
  assert(Literal.isAscii() && "Didn't allow wide strings in");
#else
  assert(Literal.isOrdinary() && "Didn't allow wide strings in");
#endif

  if (Literal.hadError)
    return false;

  if (Literal.Pascal) {
    Diag(StrToks[0].getLocation(), diag::err_expected_string_literal)
      << /*Source='in...'*/0 << DiagnosticTag;
    return false;
  }

  String = std::string(Literal.GetString());
  return true;
 }
}