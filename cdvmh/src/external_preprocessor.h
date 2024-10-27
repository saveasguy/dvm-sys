//===----------------------------------------------------------------------===//
// This file was originally written for SAPFOR, it also uses some code from
// llvm/Lex/Preprocessor.cpp to reimplement necessary functions here.
//===----------------------------------------------------------------------===//

#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/Lex/Preprocessor.h>
#include <llvm/ADT/ArrayRef.h>

namespace cdvmh {
/// This is similar to preprocessor but allows us reprocess already lexed tokens
/// in a simple way. Note, that since a LLVM 9.0.0 it is not possible to enable
/// backtracking inside a call of the Preprocessor::Lex().
///
/// TODO: ExternalPreprocessor has not able to distinct macro yet.
/// Hence, if ExternalPreprocessor is built from unexpanded token stream,
/// it only works with unexpanded tokens and disables macro expansion.
/// Otherwise, all tokens have been already expanded outside this preprocessor.
class ExternalPreprocessor {
  using TokenStream = llvm::SmallVector<clang::Token, 64>;
public:
  ExternalPreprocessor(clang::Preprocessor &PP,
      llvm::ArrayRef<clang::Token> Toks)
    : mPP(&PP)
    , mTokens(Toks.begin(), Toks.end())
    , mNextPtr(mTokens.begin())
    , mEndOfStream(mTokens.end()) {}

  /// Lex the next token for this preprocessor.
  bool Lex(clang::Token &Tok) {
    if (mNextPtr == mEndOfStream)
      return false;
    Tok = *mNextPtr;
    ++mNextPtr;
    return true;
  }

  /// TODO: ExternalPreprocessor has not able to distinct macro yet.
  void LexUnexpandedToken(clang::Token &Result) {
    Lex(Result);
  }

  /// Lex a token.  If it's a comment, keep lexing until we get
  /// something not a comment.
  ///
  /// This is useful in -E -C mode where comments would foul up preprocessor
  /// directive handling.
  void LexNonComment(clang::Token &Result) {
    do
      Lex(Result);
    while (Result.getKind() == clang::tok::comment);
  }


  /// Insert tokens at the current position and start lexing tokens from
  /// the first inserted tokens. The current position is relexing when
  /// processing of knew tokens is finished.
  void EnterTokenStream(llvm::ArrayRef<clang::Token> Toks) {
    if (mNextPtr == mTokens.begin())
      mNextPtr = mTokens.insert(mNextPtr, Toks.begin(), Toks.end());
    else
      mNextPtr = mTokens.insert(mNextPtr - 1, Toks.begin(), Toks.end());
    mEndOfStream = mTokens.end();
  }

  /// True if EnableBacktrackAtThisPos() was called and
  /// caching of tokens is on.
  bool isBacktrackEnabled() const { return !mBacktrackPositions.empty(); }

  void EnableBacktrackAtThisPos() {
    mBacktrackPositions.push_back(mNextPtr);
  }

  void CommitBacktrackedTokens() {
    assert(!mBacktrackPositions.empty() &&
           "EnableBacktrackAtThisPos was not called!");
    mBacktrackPositions.pop_back();
  }

  /// Make Preprocessor re-lex the tokens that were lexed since
  /// EnableBacktrackAtThisPos() was previously called.
  void Backtrack() {
    assert(!mBacktrackPositions.empty() &&
           "EnableBacktrackAtThisPos was not called!");
    mNextPtr = mBacktrackPositions.pop_back_val();
  }

  /// When backtracking is enabled and tokens are cached,
  /// this allows to revert a specific number of tokens.
  ///
  /// Note that the number of tokens being reverted should be up to the last
  /// backtrack position, not more.
  void RevertCachedTokens(unsigned N) {
    assert(isBacktrackEnabled() &&
           "Should only be called when tokens are cached for backtracking!");
    assert(unsigned(mNextPtr - mBacktrackPositions.back()) >= N
         && "Should revert tokens up to the last backtrack position, not more!");
    mNextPtr -= N;
  }

  /// Parses a simple integer literal to get its numeric value.
  ///
  /// Floating point literals and user defined literals are rejected.
  /// Used primarily to handle pragmas that accept integer arguments.
  bool parseSimpleIntegerLiteral(clang::Token &Tok, uint64_t &Value);

	/// Complete the lexing of a string literal where the first token has already
  /// been lexed.
  bool FinishLexStringLiteral(clang::Token &Result, std::string &String,
      const char *DiagnosticTag, bool AllowMacroExpansion);

  /// CreateString - Plop the specified string into a scratch buffer and return a
  /// location for it.  If specified, the source location provides a source
  /// location for the token.
  void CreateString(llvm::StringRef Str, clang::Token &Tok,
                    clang::SourceLocation ExpansionLocStart,
                    clang::SourceLocation ExpansionLocEnd) {
    mPP->CreateString(Str, Tok, ExpansionLocStart, ExpansionLocStart);
  }

  clang::DiagnosticBuilder Diag(clang::SourceLocation Loc,
      unsigned DiagID) const {
    return mPP->Diag(Loc, DiagID);
  }

  clang::DiagnosticBuilder Diag(const clang::Token &Tok,
      unsigned DiagID) const {
    return mPP->Diag(Tok.getLocation(), DiagID);
  }

  clang::DiagnosticsEngine &getDiagnostics() const {
    return mPP->getDiagnostics();
  }
  const clang::LangOptions &getLangOpts() const { return mPP->getLangOpts(); }

  const clang::TargetInfo &getTargetInfo() const {
    return mPP->getTargetInfo();
  }

  clang::SourceManager &getSourceManager() const {
    return mPP->getSourceManager();
  }

  clang::FileManager &getFileManager() const { return mPP->getFileManager(); }

  /// Return the 'spelling' of the Tok token.
  std::string getSpelling(const clang::Token &Tok,
      bool *Invalid = nullptr) const {
    return mPP->getSpelling(Tok, Invalid);
  }

  /// Get the spelling of a token into a SmallVector.
  llvm::StringRef getSpelling(const clang::Token &Tok,
      llvm::SmallVectorImpl<char> &Buffer, bool *Invalid) const {
    return mPP->getSpelling(Tok, Buffer, Invalid);
  }

  /// Get the spelling of a token into a preallocated buffer,
  /// instead of as an std::string.
  unsigned getSpelling(const clang::Token &Tok, const char *&Buffer,
      bool *Invalid = nullptr) const {
    return mPP->getSpelling(Tok, Buffer, Invalid);
  }

private:
  clang::Preprocessor *mPP;
  TokenStream mTokens;
  clang::Token *mNextPtr;
  clang::Token *mEndOfStream;
  llvm::SmallVector<clang::Token *, 8> mBacktrackPositions;
};
}
