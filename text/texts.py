#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:07:56 2021

@author: alain
"""

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# In order to perform text animation, need to format text with basic
# instruction: paragraph formatting, justification,....

# ---------------------------------------------------------------------------
# Text item (either the whole text, words or chars depending on the needs)
# Provide the text structure in a tree mode to compute the formatting
#
# Provides two arrays which can be used by the text object:
# - texts : the linear list of text items
# - sizes : the base widths (and heights) of the text items

import numpy as np

class Token():
    
    SPACE  = 0x01
    EOL    = 0x02
    LETTER = 0x04
    FIGURE = 0x08
    PUNC   = 0x10
    DECO   = 0x20
    EOF    = 0x40
    
    IN_WORD   = LETTER ^ FIGURE
    IN_FIGURE = FIGURE
    
    def __init__(self, text, owner=None):
        
        # Text structure
        self.text     = text
        self.owner    = owner
        self.children = None
        if self.owner is not None:
            owner.add(self)
            
        # An item can be followed by eols and spaces
        self.spaces_after = 0 # Number of spaces after
        self.eols_after   = 0 # Number of eols after
        
        # Will be computed afterwhile
        #self.width_ = 0.  # Width for leaf tokens
        self.after_ = 0   # Trailer space (used with adjencent tokens, ignored when at the end)
        
    # ---------------------------------------------------------------------------
    # Pretty representation
    
    def __repr__(self):
        cn = 0 if self.children is None else len(self.children)
        
        s = f"<Token('{self.text}') is_leaf: {self.is_leaf}, children: {cn}, total leafs: {self.total_leafs}\n"
        s += f"[{self.whole_text}]>"
        return s

    # ---------------------------------------------------------------------------
    # Add a children in the list
        
    def add(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
            
    # ---------------------------------------------------------------------------
    # Navigating in the tree structure
            
    @property
    def is_top(self):
        return self.owner is None
    
    @property
    def top(self):
        if self.owner is None:
            return self
        else:
            return self.owner.top
        
    @property
    def is_leaf(self):
        return self.children is None
    
    @property
    def depth(self):
        if self.owner is None:
            return 0
        else:
            return 1 + self.owner.depth
        
    @property
    def is_word(self):
        return self.depth == 1
    
    @property
    def is_char(self):
        return self.depth == 2
    
    @property
    def total_leafs(self):
        if self.is_leaf:
            return 1
        else:
            c = 0
            for child in self.children:
                c += child.total_leafs
            return c
        
    @property
    def chars(self):
        if self.is_leaf:
            return [self.text]
        else:
            cs = []
            for child in self.children:
                cs.extend(child.chars)
            return cs
        
    def dump(self):
        print("   "*self.depth, f"{self.width:7.1f} (+{self.after:3.1f}):", self.text)
        if not self.is_leaf:
            for child in self.children:
                child.dump()
        
    # ---------------------------------------------------------------------------
    # Character type
    
    @staticmethod
    def char_type(c):
        
        code = ord(c)
        
        if c == ' ':
            return Token.SPACE
        elif c == '\n':
            return Token.EOL
        elif c == "-":
            return Token.MINUS
        elif c == '_':
            return Token.LETTER
        elif c in [',', '.', ':', ';', '!', '?']:
            return Token.PUNC
        
        elif code < ord("0"):
            return Token.PUNCT
        
        elif code <= ord("9"):
            return Token.FIGURE
        
        elif code < ord('A'):
            return Token.PUNCT
        
        elif code <= ord('Z'):
            return Token.LETTER
        
        elif code < ord('a'):
            return Token.PUNCT
        
        elif code < ord('z'):
            return Token.LETTER
        
        elif code <= 128:
            return Token.PUNCT
        else:
            return Token.LETTER
        
    # ---------------------------------------------------------------------------
    # Split the text in words
        
    def split_in_words(self, chars=True):
        
        class Reader():
            def __init__(self, text):
                self.text = text
                self.offset = 0
                self.cmax = len(self.text)+1
                
            def read_char(self):
                if self.offset >= len(self.text):
                    return 0x00, Token.EOF
                else:
                    c = self.text[self.offset]
                    self.offset += 1
                    return c, Token.char_type(c)
                
            def next_char(self):
                c, ct = self.read_char()
                self.offset -= 1
                return c, ct
            
            def back(self, ctype_read):
                if ctype_read != Token.EOF:
                    self.offset -= 1
                
            def read_word(self):
                word = ""
                for i in range(self.cmax):
                    c, ct = self.read_char()
                    
                    if ct & Token.IN_WORD:
                        word += c
                    elif ct & Token.PUNC:
                        word += c
                        return word
                    else:
                        self.back(ct)
                        return word

            def read_number(self):
                s = ""
                for i in range(self.cmax):
                    c, ct = self.read_char()
                    s += c
                    try:
                        _ = float(s)
                    except:
                        if ct & Token.PUNC:
                            return s
                        else:
                            self.back(ct)
                            return s[:-1]
                    
            def spaces_count(self):
                count = 0
                for i in range(self.cmax):
                    c, ct = self.read_char()
                    if c == ' ':
                        count += 1
                    else:
                        self.back(ct)
                        return count
                    
            def eols_count(self):
                count = 0
                for i in range(self.cmax):
                    c, ct = self.read_char()
                    if c == '\n':
                        count += 1
                        self.spaces_count()
                    else:
                        self.back(ct)
                        return count
                    
            def read_after(self, token):
                token.spaces_after = reader.spaces_count()
                token.eols_after   = reader.eols_count()
                return token
        
        # ----- The characters reader
        
        reader = Reader(self.text)
        
        for i in range(reader.cmax):
            
            c, ctype = reader.read_char()
            
            if ctype == Token.EOF:
                break
            
            elif ctype & Token.LETTER:
                reader.back(ctype)
                word  = reader.read_word()
                token = reader.read_after(Token(word, self))
                
            elif ctype & Token.FIGURE:
                reader.back(ctype)
                word  = reader.read_number()
                token = reader.read_after(Token(word, self))
                
            else:
                token = reader.read_after(Token(c, self))
                
            if chars:
                token.split_in_chars()
                
    # ---------------------------------------------------------------------------
    # Split the text in chars
        
    def split_in_chars(self):
        if self.is_leaf:
            for c in self.text:
                Token(c, self)
        else:
            for child in self.children:
                child.split_in_chars()
        
    # ---------------------------------------------------------------------------
    # The original text can be rebuilt from the structure        
        
    @property
    def whole_text(self):
        
        if self.is_leaf:
            s = self.text
        else:
            s = ""
            for child in self.children:
                s += child.whole_text
                
        # Add spaces and eols 
        
        s += ' '*self.spaces_after
        s += '\n'*self.eols_after
        
        # Return the whole text
        
        return s
        
    # ---------------------------------------------------------------------------
    # The widths are computed with a metrics width the following attributes:
    # - space_width      : width of space character
    # - line_height      : vertical space between lines
    # - char_wa(char)    : width plus extra space after
    # - text_width(text) : width of a piece of text
    
    def compute_widths(self, metrics):
        
        if self.is_top:
            self.line_height = metrics.line_height
        
        self.after = 0.
        
        if self.is_leaf:
            if len(self.text) == 1:
                self.width_, self.after = metrics.char_wa(self.text)
            else:
                self.width_ = metrics.text_width(self.text)
        else:
            for child in self.children:
                child.compute_widths(metrics)
                
        if self.is_word:
            self.after  = self.spaces_after * metrics.space_width
    
    # ---------------------------------------------------------------------------
    # The width of an item is either its own width or the sum
    # of the widths of its children        
        
    @property
    def width(self):
        
        if self.is_leaf:
            
            return self.width_
        
        else:
            w = 0.
            for i, child in enumerate(self.children):
                w += child.width
                if i < len(self.children)-1:
                    w += child.after
            return w
        
    # ---------------------------------------------------------------------------
    # Paragraphes are range in the words
    
    @property
    def paragraphs(self):

        if self.is_leaf:
            return None

        ps = []
        n = len(self.children)
        i0 = 0
        for i in range(n):
            if (self.children[i].eols_after > 0) or (i == n-1):
                ps.append((i0, i+1))
                i0 = i+1
                
        return ps
        
    # ---------------------------------------------------------------------------
    # Format a list of tokens within a given length
    # Possible behaviors on on_over
    # - CONTINUE  : continue after
    # - RESCALE   : rescale the font
    # - NEW_LINE  : change the line
    
    def align_children(self, width=None, align_x='LEFT', on_over='CONTINUE'):
        
        if self.is_leaf:
            return None
        
        # ---------------------------------------------------------------------------
        # Array with:
        # 0 : x location
        # 1 : y location
        # 2 : width
        # 3 : after (for work)
        
        xywa = np.zeros((len(self.children), 4), np.float)
        
        # ----- The token widths
        xywa[:,  2] = [child.width for child in self.children]
        
        # ----- Space after each token
        xywa[:,  3] = [child.after for child in self.children]
        
        # ----- The x comes from the cumulative sum of the widths adn afters
        if len(self.children) > 1:
            xywa[1:, 0]  = np.cumsum(xywa[:-1, 2] + xywa[:-1, 3])
            
        # ----- No width : done
        if width is None and (self. is_leaf):
            return xywa[:, :3]
            
        # ----- Compute the width between to indices
        
        def comp_token_width(i0, i1):
            
            # Location + width without after

            return xywa[i1-1, 0] + xywa[i-1, 2] - xywa[i0, 0]
        
        # ----- Loop on the paragraphs
        
        paragraphs = self.paragraphs
        y = 0
        
        for i_start, i_end in paragraphs:
            
            # ----- Do we have to split lines
            i0 = i_start

            for i in range(1000): # Exit with break

                # ----- All the tokens to left on the correct line
                
                xywa[i0:, 0] -= xywa[i0, 0]
                xywa[i0:, 1] = y
        
                # ----- Let's find a range which is shorter than the constraint
                
                if (width is not None) and (on_over == 'NEW_LINE'):
                    
                    i1 = i0 + 1 # At least one more token !
                    for i in reversed(range(i0+1, i_end+1)):
                        if comp_token_width(i0, i) < width:
                            i1 = i
                            break
                else:
                    i1 = i_end
                    
                # The current token width
                token_width = comp_token_width(i0, i1)
            
                # The difference
                if width is None:
                    diff = 0
                else:
                    diff = width - token_width
                    
                # ----- Only if the difference is positive
                
                if diff > 0.0001:
    
                    # ----- The difference is positive: we can do it
                        
                    align = align_x
                    if align == 'JUSTIFY' and ((i1 == i_end) and not self.is_word):
                        align = 'LEFT'
    
                    if align == 'CENTER':
                        
                        xywa[i0:i1, 0] += diff / 2
                        
                    elif align == 'RIGHT':
                        
                        xywa[i0:i1, 0] += diff
                        
                    elif align == 'JUSTIFY':

                        ntk = i1 - i0
                        
                        # ----- We are on a word : we can simply change
                        # the locations of the characters
                        
                        if self.is_word:

                            if ntk == 1:
                                xywa[i0:i1, 0] += diff / 2
                            else:
                                xywa[i0:i1, 0] += np.arange(ntk)*diff/(ntk - 1)
                                
                        # ----- We are on a text : we can also change
                        # the width of the words
                        
                        else:
                            if ntk == 1:
                                xywa[i0:i1, 2] = width
                            else:
                                r = width / token_width
                                xywa[i0:i1, [0, 2]] *= r
                            
                # ----- New line
                if self.is_top:
                    y -= self.line_height

                # -----Time to exit the paragraph loop ?
                if i1 == i_end:

                    # Supplemental eols perhaps
                    if self.children[i1-1].eols_after > 1:
                        y -= self.line_height*(self.children[i1-1].eols_after - 1)

                    break # Next paragraph
                else:
                    
                    # Let's continue to loop on the paragraph
                    i0 = i1
                    
        # ----------------------------------------------------------------------------------------------------
        # The words are alignes
        # Let's align the chars
        
        if True and self.is_top:
            full_xyw = None
            
            align = 'JUSTIFY' if align_x == 'JUSTIFY' else 'LEFT'
            
            for i, child in enumerate(self.children):
                child_xyw = child.align_children(width=xywa[i, 2], align_x=align, on_over='CONTINUE')
                child_xyw[:, 0] += xywa[i, 0]
                child_xyw[:, 1] = xywa[i, 1]
                
                if full_xyw is None:
                    full_xyw = child_xyw
                else:
                    full_xyw = np.append(full_xyw, child_xyw, axis=0)
                    
            return full_xyw
    
        else:
            
            return xywa[:, :3]
        


def tests():
    
    class Metrics():
        @property
        def space_width(self):
            return 1
        
        def char_width(self, c):
            return 10
    
        def char_after(self, c):
            return 0.1
        
    words = Token("Salut les 123.45 enfants! "*2)
    words.split_in_words(True)
    words.compute_widths(Metrics())
    
    print(words)
    for child in words.children:
        print(child.text)
    #words.dump()
    #print("Paragraphs:", words.paragraphs)
        
    def test(lg=0):
        line = xywa[xywa[:, 1] == lg]
        print("----- Line", lg)
        print(lg, line[:, 0])
        print(lg, line[:, 1])
        print(lg, line[:, 2])
        print(line[-1, 0] + line[-1, 2])
        print()
    
    xywa = words.align_children(width=150, align_x='JUSTIFY', on_over='NEW_LINE')
    for i in range(len(xywa)):
        print(f"{int(xywa[i, 1]):2d}: {xywa[i, 0]:5.1f}  {xywa[i, 2]:5.1f} --> {(xywa[i, 0] + xywa[i, 2]):5.1f}")
    
    print()
    test(lg=0)
    test(lg=1)
    test(lg=2)
    test(lg=3)
    test(lg=4)



