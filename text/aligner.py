#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:17:37 2021

@author: alain
"""

import numpy as np

# =============================================================================================================================
# Elementary alignment of a series of tokens given their width and their spacing
# It splits the series in lines and place the tokens per line according the alignment instruction
# A token can be resizable or not

def align(token_wa, width=None, align_x='LEFT', variable_widths=False, keep_with_next=None):
    """Align the tokens according alignment instruction.
    
    Parameters
    ----------
    token_wa : array (n, 2) of floats
        The width of each token and the space to keep after the token when it is not the last of the line.
    width : float, optional
        The line width into which aligning the tokens. The default is None.
    align_x : str, optional
        A valid alignement code in ['LEFT', 'CENTER', 'RIGHT', 'JUSTIFY']. The default is 'LEFT'.
    variable_widths: bool or array of bools, optinoal
        For JUSTIFY alignment, tells if the width of each tokenn can be changed.
    keep_with_next: array of bools, optional
        Indicate the tokens which must be linked with the next one

    Returns
    -------
    array (, 3) of floats
        location (x, y) and width of the tokens (width is the same as the inputif not changed in JUSTIFY alignment)
    """
    
    # ---------------------------------------------------------------------------
    # The result array with:
    # 0 : x location
    # 1 : y location
    # 2 : width
    
    # For source code clarity: xyw indices
    X = 0
    Y = 1
    W = 2
    
    count = len(token_wa)
    
    xyw = np.zeros((count, 3), np.float)
    
    # ----- The token widths
    xyw[:,  W] = token_wa[:, 0]

    # ----- The x comes from the cumulative sum of the widths ann spaces after
    xyw[1:, X]  = np.cumsum(token_wa[:-1, 0] + token_wa[:-1, 1])
        
    # ----- If no width, we are done
    
    if count == 0 or (width is None):
        return xyw
    
    # --------------------------------------------------
    # Adjust keep
    
    def keep_next(index):
        
        if keep_with_next is None:
            return False
        
        if index == count:
            return False
        
        return keep_with_next[index]
    
    def next_index(index):
        if keep_next(index):
            return index+2
        else:
            return index+1
        
    # --------------------------------------------------
    # Variable width
    
    def var_width(index):
        if hasattr(variable_widths, '__len__'):
            return variable_widths[index]
        else:
            return variable_widths
    
    # --------------------------------------------------
    # Utility function : compute the width between two indices
    # Location + width without after
    # Interval is [i0, i1[
    
    def comp_interval_width(i0, i1):
        return xyw[i1-1, X] + xyw[i1-1, W] - xyw[i0, X]
    
    # --------------------------------------------------
    # ----- Splitting loop
    
    start_index = 0
    y = 0
    while start_index < count:
        
        # ----- The remaining tokens starting from 0 in the line
        
        xyw[start_index:, X] -= xyw[start_index, X]
        xyw[start_index:, Y] = y

        # ----- Let's find a range which is shorter than the constraint
        
        last_index = next_index(start_index)
        while last_index < count:
            
            if comp_interval_width(start_index, next_index(last_index)) > width+0.001:
                break
            
            last_index = next_index(last_index)
            
        # ----- The width of the tokens interval

        line_width = comp_interval_width(start_index, last_index)
    
        # ----- Difference with the target width
        
        if width is None:
            diff = 0
        else:
            diff = width - line_width
            
        # ----- If the difference is positive, we can take
        # the alignement into account
        
        if diff > 0:
            
            # We don't justify the last line
            align = align_x
            if (align == 'JUSTIFY') and (last_index == count) and variable_widths:
                align = 'LEFT'
                
            # ----- CENTER and RIGHT are simple

            if align == 'CENTER':
                
                xyw[start_index:last_index, X] += diff / 2
                
            elif align == 'RIGHT':
                
                xyw[start_index:last_index, X] += diff
                
            # ----- JUSTIFY will depend upon we can change the width of the tokens
                
            elif align == 'JUSTIFY':

                # Number of tokens in the interval
                
                ntk = last_index - start_index
                
                # One single token
                if ntk == 1:
                    if var_width(start_index):
                        xyw[start_index, W] = width
                        
                # Several tokens
                else:
                    
                    # Avaiable spaces
                    spaces = xyw[start_index + 1:last_index, X] - xyw[start_index:last_index-1, X] - xyw[start_index:last_index-1, W]
                        
                    # Variable widths
                    widths = np.array([xyw[i, W] if var_width(i) else 0. for i in range(start_index, last_index)])
                    
                    # Total resizable
                    length     = xyw[last_index-1, X] + xyw[last_index-1, W]
                    var_length = np.sum(spaces) + np.sum(widths)
                    fix_length = length - var_length
                    
                    # ---------------------------------------------------------------------------
                    # Ratio such as : (var_length) * ratio + fixe_length = width

                    r = (width - fix_length) / var_length
                    
                    # New spaces
                    spaces *= r
                    
                    # Let's resize the resizable items
                    for index in range(start_index, last_index):
                        
                        # New width is resizable
                        if var_width(index):
                            xyw[index, W] *= r
                            
                        # Update the space
                        if index > start_index:
                            xyw[index, X] = xyw[index-1, X] + xyw[index-1, W] + spaces[index-start_index-1]
                            
                        
        # ---------------------------------------------------------------------------
        # Time the create a new line
        
        y -= 1
        start_index = last_index
        
    # ----------------------------------------------------------------------------------------------------
    # We return the locations and widths
    
    return xyw

# =============================================================================================================================
# Number of line in a xyw

def lines_count(xyw):
    count = 1
    for i in range(1, len(xyw)):
        if xyw[i, 1] != xyw[i-1, 1]:
            count += 1
            
    return count

# =============================================================================================================================
# An iterator of the lines of a xyw structure 

class lines_iter():
    def __init__(self, xyw):
        self.xyw      = xyw
        self.cur_line = 0
        
    def __iter__(self):
        self.cur_line = 0
        return self
    
    def __next__(self):
        
        line = []
        n = len(self.xyw)
        
        if self.cur_line >= n:
            raise StopIteration
            
        cur = self.cur_line
        self.cur_line = n
        for i in range(cur, n):
            if self.xyw[i, 1] == self.xyw[cur, 1]:
                line.append(i)
            else:
                self.cur_line = i
                break
            
        return line
            
# =============================================================================================================================
# Vertical alignment uses line height

def vertical_align(xyw, height=None, align_y='TOP', line_height=1.):
    
    xyw[:, 1] *= line_height
    text_height = -np.min(xyw[:, 1]) + line_height
    
    if height is None:
        return xyw
    
    diff = height - text_height
    
    if diff > 0:
    
        if align_y == 'CENTER':
            text_height[:, 1] -= diff / 2
            
        elif align_y in ['BOT', 'BOTTOM']:
            text_height[:, 1] -= diff
            
        elif align_y == 'JUSTIFY':
            
            lc = lines_count(xyw)
            if lc > 1:
                delta = diff / (lc-1)
                lines = [line for line in lines_iter(xyw)]
                for i, line in enumerate(lines):
                    xyw[line, 1] -= delta*i
    
    return xyw


# =============================================================================================================================
# A utility cmlass to read a text as a flow of chars

class TextReader():
    
    # Chars types
    
    SPACE  = 0x01
    EOL    = 0x02
    LETTER = 0x04
    FIGURE = 0x08
    PUNCT  = 0x10
    DECO   = 0x20
    EOF    = 0x40
    
    IN_WORD   = LETTER ^ FIGURE
    IN_FIGURE = FIGURE
    
    
    def __init__(self, text):
        self.text = text
        self.offset = 0
        self.cmax = len(self.text)+1
        
    # ----------------------------------------------------------------------------------------------------
    # Character type
    
    @staticmethod
    def char_type(c):
        
        code = ord(c)
        
        if c == ' ':
            return TextReader.SPACE
        elif c == '\n':
            return TextReader.EOL
        elif c == "-":
            return TextReader.MINUS
        elif c == '_':
            return TextReader.LETTER
        elif c in [',', '.', ':', ';', '!', '?']:
            return TextReader.PUNCT
        
        elif code < ord("0"):
            return TextReader.DECO
        
        elif code <= ord("9"):
            return TextReader.FIGURE
        
        elif code < ord('A'):
            return TextReader.DECO
        
        elif code <= ord('Z'):
            return TextReader.LETTER
        
        elif code < ord('a'):
            return TextReader.DECO
        
        elif code < ord('z'):
            return TextReader.LETTER
        
        elif code <= 128:
            return TextReader.DECO
        else:
            return TextReader.LETTER
        
        
    def read_char(self):
        if self.offset >= len(self.text):
            return 0x00, TextReader.EOF
        else:
            c = self.text[self.offset]
            self.offset += 1
            return c, TextReader.char_type(c)
        
    def next_char(self):
        c, ct = self.read_char()
        self.offset -= 1
        return c, ct
    
    def back(self, ctype_read):
        if ctype_read != TextReader.EOF:
            self.offset -= 1
        
    def read_word(self):
        word = ""
        for i in range(self.cmax):
            c, ct = self.read_char()
            
            if ct & TextReader.IN_WORD:
                word += c
            #elif ct & TextReader.PUNCT:
            #    word += c
            #    return word
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
                if ct & TextReader.PUNCT:
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
            
# =============================================================================================================================
# A token of a formatted text
#
# A token is either the whole text, a word or a character

class Token():
    
    CHAR = 0
    WORD = 1
    PARA = 2
    TEXT = 3
    
    def __init__(self, text, nature=TEXT):
        
        # Text content
        self.text     = text
        
        # Node structure
        self.owner    = None
        self.children = None
        
        # nature
        self.nature = nature
        if nature == Token.WORD:
            self.spaces_after = 0
        
        # For dimensions
        self.width = 0
        self.after = 0

    # ---------------------------------------------------------------------------
    # Pretty representation
    
    def __repr__(self):
        cn = 0 if self.children is None else len(self.children)
        
        s = f"<{self.nature_label} Token ('{self.text}') is_leaf: {self.is_leaf}, children: {cn}, total leafs: {self.leafs_count}\n"
        s += f"[{self.whole_text}]>"
        return s
    
    @property
    def nature_label(self):
        return ['Character', 'Word', 'Paragraph', 'Text'][self.nature]
    
    def __len__(self):
        return 0 if self.children is None else len(self.children)
    
    def __getitem__(self, index):
        return self.children[index]

    # ---------------------------------------------------------------------------
    # Add a child in the list
        
    def add(self, child):
        
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
        child.owner = self
        
        return child
    
    # ---------------------------------------------------------------------------
    # Insert a child
    
    def insert_after(self, after, child):
        children = []
        for ch in self.children:
            children.append(ch)
            if ch == after:
                children.append(child)
        self.children = children
        return child
    
    # ---------------------------------------------------------------------------
    # Merge with next
    
    def merge_with_next(self):
        
        ntok = self.follower
        if ntok is None:
            raise RuntimeError(f"The text token '{self.text}' in '{self.owner.text}' has no follower")
            
        if ntok.children is not None:
            for child in ntok.children:
                self.add(child)
                
        ntok.children = None
        self.text += ntok.text
        
        children = []
        for child in self.owner.children:
            if child != ntok:
                children.append(child)
        self.owner.children = children
        
        del ntok
    
    # ---------------------------------------------------------------------------
    # Split a token at a given index
    
    def split_at(self, at_index):
        
        if at_index >= len(self.text):
            raise RuntimeError(f"Text token '{self.text}' can't be splitten at index {at_index}")
            
        text1 = self.text[:at_index]
        text2 = self.text[at_index:]
        
        token = self.owner.insert_after(self, Token(text2, self.nature))
        for i in range(at_index, len(self.children)):
            token.add(self.children[i])
            
        self.children = self.children[:at_index]
        self.text     = text1
        
        return token
            
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Nodes tree implementation
            
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
    def leafs_count(self):
        if self.is_leaf:
            return 1
        else:
            c = 0
            for child in self.children:
                c += child.leafs_count
            return c
        
    @property
    def index_in_owner(self):

        if self.owner is None:
            return None
        
        for i, child in enumerate(self.owner.children):
            if child == self:
                return i
            
        raise RuntimeError("Token tree structure is inconsistent :-(")
        
    @property
    def follower(self):
        
        if self.owner is None:
            return None
        
        index = self.index_in_owner
        
        if index == len(self.owner.children)-1:
            return None
        else:
            return self.owner.children[index+1]
        
    @property
    def chars_count(self):
        
        if self.nature == Token.CHAR:
            return 1
        
        elif self.children is not None:
            count = 0
            for child in self.children:
                count += child.chars_count
            return count
        
        else:
            return 0
        
    @property
    def is_word(self):
        return self.nature == Token.WORD and (TextReader.char_type(self.text[0]) & TextReader.IN_WORD != 0)

    @property
    def is_punctuation(self):
        return self.nature == Token.WORD and (TextReader.char_type(self.text[0]) & TextReader.PUNCT != 0)

    @property
    def keep_with_next(self):
        if self.is_word:
            foll = self.follower
            if foll is not None:
                return foll.is_punctuation
        return False

    # ---------------------------------------------------------------------------
    # Loop on the nodes
        
    def foreach(self, f, leafs_only=False, children_before=False, nature=None):
        
        # ------ If nature is specified, the other arguments are irrelevant
        
        if nature is not None:
            if self.nature == nature:
                return f(self)
            
            elif self.children is not None:
                for child in self.children:
                    r = child.foreach(f, nature=nature)
                    if r is not None:
                        return r
                    
            return None

        # ------ Other arguments
        
        if self.is_leaf:
            
            return f(self)
        
        else:
            if (not leafs_only) and (not children_before):
                r = f(self)
                if r is not None:
                    return r

            for child in self.children:
                r = child.foreach(f, leafs_only=leafs_only, children_before=children_before)
                if r is not None:
                    return r
                
            if (not leafs_only) and children_before:
                r = f(self)
                if r is not None:
                    return r
                
            return None
        
    # ---------------------------------------------------------------------------
    # The tokens by nature
    # 
    
    def tokens(self, select=None):
        tokens = []

        def f(token):
            
            if select is None:
                tokens.append(token)
                
            elif type(select) is int:
                if token.nature == select:
                    tokens.append(token)
                    
            elif type(select) is str:
                if token.text == select:
                    tokens.append(token)
                    
            elif hasattr(select, '__call__'):
                if select(token):
                    tokens.append(token)
                    
            else:
                raise RuntimeError(f"Token selection error: unknown selection '{select}'")
            
        self.foreach(f)
        
        return tokens
        
        
    # ---------------------------------------------------------------------------
    # The array of texts
    
    def texts(self, nature):
        texts = []

        def f(token):
            texts.append(token.text)
            
        self.foreach(f, nature=nature)
        
        return texts

    # ---------------------------------------------------------------------------
    # The array of chars
    
    @property
    def chars(self):
        return self.texts(Token.CHAR)

    # ---------------------------------------------------------------------------
    # The array of words
    
    @property
    def words(self):
        return self.texts(Token.WORD)
    
    # ----------------------------------------------------------------------------------------------------
    # Build the structure of index
    
    def index_structure(self, para_index=0, word_index=0, char_index=0):
        
        self.para_index = para_index
        self.word_index = word_index
        self.char_index = char_index
        
        if self.nature == Token.CHAR:
            return para_index, word_index, char_index+1
        
        for child in self.children:
            para_index, word_index, char_index = child.index_structure(para_index, word_index, char_index)
            
        if self.nature == Token.WORD:
            return para_index, word_index+1, char_index
        
        elif self.nature == Token.PARA:
            return para_index+1, word_index, char_index
                
    
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
        
        if hasattr(self, "spaces_after"): s += ' '*self.spaces_after
        if hasattr(self, "eols_after"):   s += '\n'*self.eols_after
        
        # Return the whole text
        
        return s
        
    def dump(self):
        
        def f(token):
            print("    "*token.depth, f"{token.nature_label} - '{token.text}'. {len(token)} child(ren):")
            
        self.foreach(f)
                
    # ====================================================================================================
    # Splitting methods

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Split the text in paragraphs
        
    def split_in_paragraphs(self):

        if self.nature != Token.TEXT:
            raise RuntimeError("Text splitting error: only texts can be splitten in paragraphs")
        
        del self.children
        self.children = []
        reader = TextReader(self.text)
        
        para = ""
        for i in range(reader.cmax):
            
            c, ctype = reader.read_char()
            if ctype == TextReader.EOF:
                if para != "":
                    self.add(Token(para, Token.PARA))
                break
                
            elif ctype == TextReader.EOL:
                
                token = self.add(Token(para, Token.PARA))
                token.eols_after = 1 + reader.eols_count()
                
                para = ""
                
            else:
                para += c
        
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Split the text in chars
    # The token must have been wisely initialized to avoid control chars
        
    def split_in_chars(self):
        
        if self.nature != Token.WORD:
            raise RuntimeError("Text splitting error: only words can be splitten in chars")
        
        del self.children
        self.children = []
        
        for c in self.text:
            self.add(Token(c, Token.CHAR))
            
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Split the text in words
        
    def split_in_words(self):
        
        if not self.nature in [Token.TEXT, Token.PARA]:
            raise RuntimeError("Text splitting error: only texts and paragraphs can be splitten in words")
            
        del self.children
        self.children = []
            
        reader = TextReader(self.text)
        
        for i in range(reader.cmax):
            
            c, ctype = reader.read_char()
            
            # If eols must be taken into account, use slit_in_paragraphs before
            if ctype == TextReader.EOL:
                c = " "
                ctype = TextReader.SPACE
                
            
            if ctype == TextReader.EOF:
                break
            
            elif ctype & TextReader.LETTER:
                reader.back(ctype)
                word  = reader.read_word()
                token = self.add(Token(word, Token.WORD))
                
            elif ctype & TextReader.FIGURE:
                reader.back(ctype)
                word  = reader.read_number()
                token = self.add(Token(word, Token.WORD))
                
            else:
                token = self.add(Token(c, Token.WORD))
                
            token.spaces_after = reader.spaces_count()
                
            token.split_in_chars()
            
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # For alignment, punctuation chars are kept at the end of the words
    # They can be removed afterwards
        
    def merge_punctuation(self):
        
        words = self.tokens(Token.WORD)
        index = 0
        while index < len(words)-1:
            
            word = words[index]
            foll = words[index+1]
            
            if (len(foll.text) == 1) and (word.spaces_after == 0):
                ct0 = TextReader.char_type(word.text[ 0]) & TextReader.IN_WORD
                ct1 = TextReader.char_type(word.text[-1]) & TextReader.IN_WORD
                ct2 = TextReader.char_type(foll.text[ 0]) & TextReader.PUNCT
                
                if (ct0 != 0) and (ct1 != 0) and (ct2 != 0):
                    self.spaces_after = foll.spaces_after
                    word.merge_with_next()
                    index += 1
                    
            index += 1
                    
        # Need to rebuild the index structure
        self.index_structure()
            
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # For alignment, punctuation chars are kept at the end of the words
    # They can be removed afterwards
        
    def split_punctuation(self):
        
        words = self.tokens(Token.WORD)
            
        for word in words:
            if len(word.text) > 1:
                ct0 = TextReader.char_type(word.text[ 0]) & TextReader.IN_WORD
                ct1 = TextReader.char_type(word.text[-1]) & TextReader.PUNCT
                
                if (ct0 != 0) and (ct1 != 0):
                    neww = word.split_at(len(word.text)-1)
                    neww.spaces_after = word.spaces_after
                    word.spaces_after = 0
                    
        # Need to rebuild the index structure
        self.index_structure()
            
            
    # ====================================================================================================
    # Split the text
    
    def split(self, paragraphs=True):
        
        if self.nature != Token.TEXT:
            raise RuntimeError("Text splitting error: only texts can be splitten in paragraphs and words")
            
        if paragraphs:
            self.split_in_paragraphs()
            for para in self.children:
                para.split_in_words()
                
        else:
            self.split_in_words()
            
        self.index_structure()
            

    # ====================================================================================================
    # Once splitted, the text can be formatted
    # metrics is an object with the following interface
    # - space_width      : width of space character
    # - line_height      : vertical space between lines
    # - char_width       : char width without extra space
    # - char_xwidth      : char width with the extra space
    
    def dimension_tokens(self, metrics, nature=CHAR):
        
        # ----- The size of each char
        
        def token_size(token):
            if token.nature == Token.CHAR:
                token.width = metrics.char_width(token.text)
                token.after = metrics.char_xwidth(token.text) - token.width
            else:
                token.width = 0
                for child in token.children:
                    token.width += child.width + child.after
                token.width -= token.children[-1].after
                
                if token.nature == Token.WORD:
                    token.after = metrics.space_width*token.spaces_after
            
        self.foreach(token_size, children_before=True)
        
        # ----- Return two arrays: the dimensions and the tokens
        
        tokens = self.tokens(nature)
        wa = [(token.width, token.after) for token in tokens]
        
        return tokens, np.array(wa)
    
    # ====================================================================================================
    # Align the whole text
    
    def align(self, metrics=None, width=None, align_x='LEFT', height=None, align_y='TOP'):
        
        # ----- Split the text
        
        self.split(True)
        if self.children is None:
            return [], []
        
        # ----- Dimension the tokens
        
        if metrics is not None:
            self.dimension_tokens(metrics)
        
        # ----- Get the sizes
        
        words = self.tokens(Token.WORD)
        chars = self.tokens(Token.CHAR)
        words_wa = np.array( [(word.width, word.after) for word in words] )
        chars_wa = np.array( [(char.width, char.after) for char in chars] )
        
        # ----- Paragraph formatting
        
        words_xyw = np.zeros((len(words), 3), np.float)
        y = 0
        for para in self.children:
            if para.children is not None:
                w0 = para.word_index
                w1 = w0 + len(para.children)
                words_xyw[w0:w1] = align(words_wa[w0:w1], width=width, align_x=align_x,
                            variable_widths = [len(word.text)>1 for word in words[w0:w1]],
                            keep_with_next  = [word.keep_with_next for word in words[w0:w1]])
                words_xyw[w0:w1, 1] += y
                y = np.min(words_xyw[:, 1]) - 1
                
        # ----- Vertical alignment
        
        words_xyw = vertical_align(words_xyw, height=height, align_y=align_y, line_height=metrics.line_height)

        # ----- Place the chars
        
        chars_xyw = np.zeros((len(chars), 3), np.float)
        for word, w_xyw in zip(words, words_xyw):
            ch0 = word.char_index
            ch1 = ch0 + len(word.text)
            
            xyw = align(chars_wa[ch0:ch1], width=w_xyw[2], align_x=align_x, variable_widths=False)
            xyw[:, 0] += w_xyw[0]
            xyw[:, 1]  = w_xyw[1]
            
            chars_xyw[ch0:ch1] = xyw

        # ----- Done
        
        return self.chars, chars_xyw
    
    # ====================================================================================================
    # The chars can be grouped in words, lines or paragraphs
    
    def group_chars(self, chars_xyw, target='WORDS'):
        
        # ---------------------------------------------------------------------------
        # Group by lines
        
        if target.upper() in ['LINE', 'LINES']:
            xyw = np.zeros((lines_count(chars_xyw), 3), np.float)
            for i, line in enumerate(lines_iter(chars_xyw)):
                x = chars_xyw[line[ 0], 0]
                e = chars_xyw[line[-1], 0] + chars_xyw[line[-1], 2]

                xyw[i, 0] = x
                xyw[i, 1] = chars_xyw[line[0], 1]
                xyw[i, 2] = e - x
                
            return xyw
        
        # ---------------------------------------------------------------------------
        # Group by structure
        
        if target.upper() in ['WORD', 'WORDS']:
            nature = Token.WORD
            
        elif target.upper() in ['PARA', 'PARAS', 'PARAGRAPH', 'PARAGRAPHS']:
            nature = Token.PARA
            
        elif target.upper() in ['TEXT', 'TEXTS']:
            nature = Token.TEXT
        
        else:
            raise RuntimeError(f"group_chars error: unknwon target: '{target}'")
        
        tokens = self.tokens(nature)
        xyw = np.zeros((len(tokens), 3), np.float)
        
        for index, token in enumerate(tokens):
            ch0 = token.char_index
            ch1 = ch0 + token.chars_count
            
            x = np.min(chars_xyw[ch0:ch1, 0])
            e = np.max(chars_xyw[ch0:ch1, 0]+chars_xyw[ch0:ch1, 2])
            
            xyw[index, 0] = x
            xyw[index, 1] = chars_xyw[ch0, 1]
            xyw[index, 2] = e - x
            
        return xyw

# ======================================================================================================================================================
# Test

def plot_xyw(xyw, chars=None):
    
    import matplotlib.pyplot as plt
    
    lh = 1.
    
    fig, ax = plt.subplots()
    if chars is None:
        chars = [" " for i in range(len(xyw))]
        
    for v, s in zip(xyw, chars):
        ax.plot((v[0], v[0] + v[2], v[0] + v[2], v[0], v[0]), (v[1], v[1], v[1]-lh, v[1]-lh, v[1]))
        ax.annotate(s, (v[0] + v[2]/2, v[1] - lh/2), ha='center', va='center')
        
    plt.show()

def test():
    
    s = "A very big hello! my friends : this is a test!\nPretty interesting\n"
    
    text = Token(s)
    text.split(True)
    
    def dump_struct(ttext, nature):
        tks = ttext.tokens(nature)
        for t in tks:
            print(f"{t.para_index:3d} {t.word_index:3d}{t.char_index:3d} {t.text}")

    class Metrics():
        def __init__(self):
            self.space_width = 12.
            self.line_height = 1
            np.random.seed(0)
            self.ws  = {chr(i): np.random.randint(10, 20, 1)[0] for i in range(33, 256)}
            self.xws = {chr(i): self.ws[chr(i)] + 1 for i in range(33, 256)}
            
        def char_width(self, char):
            return self.ws[char]
            return 10.
        
        def char_xwidth(self, char):
            return self.xws[char]
            return 11.
        
    align_x = 'JUSTIFY'
    chars, xyw = text.align(Metrics(), width=250, align_x=align_x)
    
    print(align_x)
    for line in lines_iter(xyw):
        x0 = xyw[line[0], 0]
        x1 = xyw[line[-1], 0] + xyw[line[-1], 2]
        l = x1-x0
        print(f"{x0:5.1f} <  {l:5.1f}  >  {x1:5.1f} :    ", end=' ')
        for i in line: print(chars[i], end=' ')
        print()
        
    plot_xyw(xyw, chars)
    
    gxyw = text.group_chars(xyw, "words")
    plot_xyw(gxyw)
    
#test()
