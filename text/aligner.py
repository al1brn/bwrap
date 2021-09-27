#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:17:37 2021

@author: alain
"""

import numpy as np

if True:
    from .glyphe import CharFormat
else:
    from glyphe import CharFormat


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
        self.text   = text
        self.offset = 0
        self.cmax   = len(self.text)+1
        
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
            return 0x00, TextReader.EOF, self.offset
        else:
            c = self.text[self.offset]
            self.offset += 1
            return c, TextReader.char_type(c), self.offset - 1
        
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
            c, ct, _ = self.read_char()
            
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
            c, ct, _ = self.read_char()
            
            if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                try:
                    _ = float(s + c)
                except:
                    self.back(ct)
                    return s[:-1]
                s += c
            else:
                self.back(ct)
                return s
            
    def spaces_count(self):
        count = 0
        for i in range(self.cmax):
            c, ct, _ = self.read_char()
            if c == ' ':
                count += 1
            else:
                self.back(ct)
                return count
            
    def eols_count(self):
        count = 0
        for i in range(self.cmax):
            c, ct, _ = self.read_char()
            if c == '\n':
                count += 1
                self.spaces_count()
            else:
                self.back(ct)
                return count
            
# =============================================================================================================================
# Manages an index or a set of indices on to the ftext.chars_ array
# Inherited by:
# - Char  : a single char
# - Word  : chars grouped in a word
# - Words : a array of words (example: text, paragraph, lines)

class FTextChars():
    
    def __init__(self, ftext, single=False):
        self.ftext  = ftext
        self.single = single
        
    # ---------------------------------------------------------------------------
    # To be overloaded
    
    @property
    def first_index(self):
        pass
    
    @property
    def last_index(self):
        pass
    
    # ---------------------------------------------------------------------------
    # Access to chars
        
    @property
    def indices(self):
        if self.single:
            return self.first_index
        else:
            return np.arange(self.first_index, self.last_index + 1)
        
    def char_at(self, index):
        return Char(self.ftext, self.index + index)
    
    @property
    def first_char_index(self):
        return self.ftext.chars_[self.first_index, 0]

    @property
    def last_char_index(self):
        return self.ftext.chars_[self.last_index, 0]
    
    @property
    def chars_count(self):
        return self.last_index - self.first_index + 1
    
    @property
    def text(self):
        return self.ftext.text_[self.first_char_index:self.last_char_index+1]
    
    @property
    def array_of_chars(self):
        return [self.ftext.text_[self.ftext.chars_[i]] for i in range(self.first_index, self.last_index+1)]

    # ---------------------------------------------------------------------------
    # Char look up
    
    def char_index(self, char, num = 0):
        for i in self.indices:
            if Char(self.ftext, i).char == char:
                if num == 0:
                    return i
                num -= 1
        return None

    # ---------------------------------------------------------------------------
    # Metrics
    
    @property
    def metrics(self):
        return self.ftext.metrics[self.indices]
        
    # ---------------------------------------------------------------------------
    # Formatting
    
    @property
    def bold(self):
        return self.ftext.metrics[self.indices, FText.BOLD]
    
    @bold.setter
    def bold(self, value):
        self.ftext.metrics[self.indices, FText.BOLD] = value
        
    @property
    def shear(self):
        return self.ftext.metrics[self.indices, FText.SHEAR]
    
    @shear.setter
    def shear(self, value):
        self.ftext.metrics[self.indices, FText.SHEAR] = value
        
    @property
    def xscale(self):
        return self.ftext.metrics[self.indices, FText.XSCALE]
    
    @xscale.setter
    def xscale(self, value):
        self.ftext.metrics[self.indices, FText.XSCALE] = value
        
    @property
    def yscale(self):
        return self.ftext.metrics[self.indices, FText.YSCALE]
    
    @yscale.setter
    def yscale(self, value):
        self.ftext.metrics[self.indices, FText.YSCALE] = value
        
    @property
    def scale(self):
        return np.array([self.xscale, self.yscale], float)
    
    @scale.setter
    def scale(self, value):
        if hasattr(value, '__len__'):
            self.xscale = value[0]
            self.yscale = value[1]
        else:
            self.xscale = value
            self.yscale = value
        
    # ---------------------------------------------------------------------------
    # Location
            
    @property
    def x(self):
        return self.ftext.metrics[self.first_index, FText.X]
    
    @x.setter
    def x(self, value):
        dx = value - self.x
        self.ftext.metrics[self.indices, FText.X] += dx

    @property
    def y(self):
        return self.ftext.metrics[self.first_index, FText.Y]
    
    @y.setter
    def y(self, value):
        self.ftext.metrics[self.indices, FText.Y] = value
        
    @property
    def location(self):
        return np.array([self.x, self.y], float)
    
    @location.setter
    def location(self, value):
        if hasattr(value, '__len__'):
            self.x = value[0]
            self.y = value[1]
        else:
            self.x = value
            self.y = value  
            
    # ---------------------------------------------------------------------------
    # Height
            
    @property
    def height(self):
        return np.max(self.ftext.metrics[self.indices, FText.HEIGHT])
            
# =============================================================================================================================
# A char is an index to the original text
# The text structure gives properties around the char

class Char(FTextChars):
    
    def __init__(self, ftext, index):
        super().__init__(ftext, single=True)
        self.index = index
        
    # ---------------------------------------------------------------------------
    # Overwrite inherited properties
    
    @property
    def first_index(self):
        return self.index
    
    @property
    def last_index(self):
        return self.index
        
    # ---------------------------------------------------------------------------
    # Display
        
    def __repr__(self):
        s = f"[Char({self.index:3d}): {self.char}] "
        return s
    
    # ---------------------------------------------------------------------------
    # Char value
    
    @property
    def char(self):
        return self.text

    # ---------------------------------------------------------------------------
    # Metrics
    
    @property
    def width(self):
        return self.ftext.metrics[self.index, FText.WIDTH]
    
    @width.setter
    def width(self, value):
        self.ftext.metrics[self.index, FText.WIDTH] = value
        
    @property
    def height(self):
        return self.ftext.metrics[self.index, FText.HEIGHT]
    
    @height.setter
    def height(self, value):
        self.ftext.metrics[self.index, FText.HEIGHT] = value
        
    @property
    def after(self):
        return self.ftext.metrics[self.index, FText.AFTER]
    
    @after.setter
    def after(self, value):
        self.ftext.metrics[self.index, FText.AFTER] = value
        
    @property
    def spaces_after(self):
        return self.ftext.chars_[self.index, 1]*self.ftext.space_width
    
    @property
    def dx(self):
        return self.ftext.metrics[self.index, FText.DX]
    
    @dx.setter
    def dx(self, value):
        self.ftext.metrics[self.index, FText.DX] = value

            
# =============================================================================================================================
# A Word is a couple (index, length) on the array of chars indices
# - chars: array of indices in the original text

class Word(FTextChars):
    
    def __init__(self, ftext, index_length):
        super().__init__(ftext, single=False)
        self.index  = index_length[0]
        self.length = index_length[1]
        
    @classmethod
    def FromIndices(cls, ftext, indices):
        return cls(ftext, [indices[0], indices[-1] - indices[0] + 1])
        
    # ---------------------------------------------------------------------------
    # Overwrite inherited properties
    
    @property
    def first_index(self):
        return self.index
    
    @property
    def last_index(self):
        return self.index + self.length - 1
        
    # ---------------------------------------------------------------------------
    # Display
        
    def __repr__(self):
        s = f"Word[{self.index:3d}, {self.length:3d}]: '{self.text}'"
        return s
    
    # ---------------------------------------------------------------------------
    # An array of Chars
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return Char(self.ftext, self.index + index)
    
    # ---------------------------------------------------------------------------
    # Chars
        
    @property
    def last_char(self):
        return self[len(self)-1]
        
    # ---------------------------------------------------------------------------
    # Metrics
    
    @property
    def width(self):
        w = 0.
        for i in range(len(self)):
            char = self[i]
            w += char.width
            if i < len(self)-1:
                w += char.after + char.spaces_after + char.dx
        return w
    
    @width.setter
    def width(self, value):
        w = 0.
        s = 0.
        for i in range(len(self)):
            char = self[i]
            w += char.width
            if i < len(self)-1:
                s += char.after + char.spaces_after
            char.dx = 0.
            
        if s == 0:
            return
                
        delta = value - w - s
        r = delta/s
        
        for i in range(len(self)):
            char = self[i]

            if i == 0:
                x = char.x
            else:
                char.x = x
            
            aft = char.after + char.spaces_after
            char.dx = r*aft
            x += char.width + aft + char.dx
        
    @property
    def after(self):
        return self.last_char.after
    
    @property
    def spaces_after(self):
        return self.last_char.spaces_after
    
    @property
    def dx(self):
        return self.last_char.dx
    
    @dx.setter
    def dx(self, value):
        self.last_char.dx = value
        
    @property
    def in_spaces(self):
        s = 0.
        for i in range(len(self)-1):
            char = self[i]
            s += char.after + char.spaces_after
        return s
    
    def zero_dx(self):
        for i in range(len(self)):
            self[i].dx = 0

# =============================================================================================================================
# A list of words can be a paragraph or a line

class Words(FTextChars):
    
    def __init__(self, ftext, words):
        super().__init__(ftext, single=False)
        self.words = words
        
    # ---------------------------------------------------------------------------
    # Overwrite inherited properties
    
    @property
    def first_index(self):
        return self[0].first_index
    
    @property
    def last_index(self):
        return self[-1].last_index
        
    # ---------------------------------------------------------------------------
    # Display
        
    def __repr__(self):
        s = f"<Words ({len(self)})\n"
        for i in range(len(self)):
            word = self[i]
            s += f"   {i:3d}: {word}\n"
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # An array of Words
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, index):
        return Word(self.ftext, self.words[index])
    
    # ---------------------------------------------------------------------------
    # Array of texts
    
    @property
    def texts(self):
        return [word.text for word in self]

    # ---------------------------------------------------------------------------
    # Word look up

    def word_indices(self, word, num=0):
        for i in range(len(self)):
            if self[i].text == word:
                if num == 0:
                    return self[i].indices
                num -= 1
        return []
    
    # ---------------------------------------------------------------------------
    # Alignment
    
    @property
    def width(self):
        w = 0.
        for i in range(len(self)):
            word = self[i]
            w += word.width
            if i < len(self)-1:
                w += word.after + word.spaces_after + word.dx
        return w
    
    @width.setter
    def width(self, value):
        w = 0.
        v = 0.
        for i in range(len(self)):
            word = self[i]
            word.zero_dx
            w += word.width
            v += word.in_spaces
            if i < len(self)-1:
                aft = word.after + word.spaces_after
                w += aft
                v += aft
                
        if v == 0.:
            return
                
        delta = value - w
        r = delta/v
        
        for i in range(len(self)):
            word = self[i]

            if i == 0:
                x = word.x
            else:
                word.x = x
                
            word.width += r*word.in_spaces

            aft = word.after + word.spaces_after
            word.dx = r*aft
            x += word.width + aft + word.dx
            
    @property
    def height(self):
        return np.max(self.ftext.metrics[self.indices, FText.HEIGHT])
        
            
# =============================================================================================================================
# The chars of a text are stored in an array
# Words, paragraphs, lines are indices to this array

class FText():
    
    WORD  = 0
    PUNCT = 1
    DECO  = 2
    
    METRICS_SIZE = 10
    
    BOLD   = 0  # Bold
    SHEAR  = 1  # Shear
    XSCALE = 2  # Scale X
    YSCALE = 3  # Scale Y
    
    X      = 4  # X location
    Y      = 5  # y location
    WIDTH  = 6  # width
    HEIGHT = 7  # height
    AFTER  = 8  # after
    DX     = 9  # DX for justifiy
    
    def __init__(self, text):
        
        self.text_ = ""
        self.text  = text
        
    def __len__(self):
        return len(self.chars_)
    
    def __getitem__(self, index):
        return Char(self, index)
    
    def __repr__(self):
        s = f"<FText '{self.text}'\n"
        s += f"paragraphs: {len(self.paras)}, words: {len(self.words)}, chars: {len(self.chars_)}, lines: {len(self.lines)}\n"
        s += "\n"
        for i_para in range(len(self.paras)):
            para = self.paras[i_para]
            s += f"Paragraph {i_para:3d}: {para}\n"
        s += "\n"
        for i_word in range(len(self.words)):
            word = self.words[i_word]
            s += f"Word {i_word:3d}: {word}\n"
        s += "\n"
        return s + ">"
    
    def reset(self):
        self.text_  = ""
        self.chars_  = np.zeros((0, 2), int) # Index in text plus number of spaces after
        self.words_  = np.zeros((0, 3), int) # (index, length) to chars plus word nature
        self.paras_  = np.zeros((0, 2), int) # (index, length) to words
        self.lines_  = np.zeros((0, 2), int) # (index, length) to words
        self.metrics = np.zeros((0, FText.METRICS_SIZE), float) 
    
    def add_char(self, index):
        self.chars_ = np.append(self.chars_, [[index, 0]], axis=0)
    
    def add_word(self, index, length, nature):
        
        char_index = len(self.chars_)
        for i in range(index, index+length):
            self.add_char(i)
        self.words_ = np.append(self.words_, [[char_index, length, nature]], axis=0)
        
        # Extend the last paragraph
        self.paras_[-1, 1] += 1
         
    # ====================================================================================================
    # Access to paragraphs, words and lines
    
    def char(self, index):
        return Char(self, index)
    
    @property
    def chars(self):
        return [Char(self, i) for i in range(len(self.chars_))]
    
    @property
    def array_of_chars(self):
        return [self.text_[i] for i in self.chars_]

    @property
    def words(self):
        return Words(self, self.words_)
    
    @property
    def paras(self):
        return [Words(self, self.words_[np.arange(para[0], para[0]+para[1])]) for para in self.paras_]
    
    @property
    def lines(self):
        return [Words(self, self.words_[np.arange(line[0], line[0]+line[1])]) for line in self.lines_]
        
    # ====================================================================================================
    # FText splitting
    
    @property
    def text(self):
        return self.text_
    
    @text.setter
    def text(self, text):
        
        self.reset()

        self.text_ = text

        reader = TextReader(text)
        
        # Create the current paragraph
        self.paras_ = np.zeros((1, 2), int)
        
        # ---------------------------------------------------------------------------
        # Loop on the chars
        # The char reader steers the progress of the char index. The for loop
        # is just here in case of algorithm bug to avoid infinite loop
        
        for wd in range(reader.cmax):
            
            # Read the current char
            
            c, ctype, index = reader.read_char()
            
            # ---------------------------------------------------------------------------
            # End of line: new paragraph
            
            if ctype == TextReader.EOL:
                para = self.paras_[-1]
                self.paras_ = np.append(self.paras_, [[para[0]+para[1], 0]], axis=0)
                continue
            
            # ---------------------------------------------------------------------------
            # End of file : done
            
            elif ctype == TextReader.EOF:
                break
            
            # ---------------------------------------------------------------------------
            # A letter : read the word
            
            elif ctype & TextReader.LETTER:
                reader.back(ctype)
                word  = reader.read_word()
                
                self.add_word(index, len(word), FText.WORD)
                
            elif ctype & TextReader.FIGURE:
                reader.back(ctype)
                word  = reader.read_number()
                
                self.add_word(index, len(word), FText.WORD)
                
            else:
                self.add_word(index, 1, FText.PUNCT if ctype == TextReader.PUNCT else FText.DECO)
                
            # ---------------------------------------------------------------------------
            # Spaces after the last read char
            
            self.chars_[-1, 1] = reader.spaces_count()
            
        # ---------------------------------------------------------------------------
        # Metrics
        
        self.init_metrics()
            
    # ====================================================================================================
    # Metrics
    
    def init_metrics(self):
        
        self.metrics = np.zeros((len(self.chars_), FText.METRICS_SIZE), float)
        self.metrics[:, FText.XSCALE] = 1.
        self.metrics[:, FText.YSCALE] = 1.
        
    # ====================================================================================================
    # metrics is an object with the following interface
    # - space_width      : width of space character
    # - char_metrics     : the metrics of the char with its format
    
    def set_metrics(self, metrics):
        
        self.space_width = metrics.space_width

        # ----- Chars matrics
        
        for index, char_index in enumerate(self.chars_[:, 0]):
            
            
            cf = CharFormat(
                    bold  = self.metrics[index, FText.BOLD],
                    shear = self.metrics[index, FText.SHEAR],
                    scale = self.metrics[index, [FText.XSCALE, FText.YSCALE]])
    
            char_metrics = metrics.char_metrics(self.text_[char_index],
                char_format = CharFormat(
                    bold  = self.metrics[index, FText.BOLD],
                    shear = self.metrics[index, FText.SHEAR],
                    scale = self.metrics[index, [FText.XSCALE, FText.YSCALE]])
                )
            
            self.metrics[index, FText.WIDTH]  = char_metrics.width
            self.metrics[index, FText.AFTER]  = char_metrics.after
            self.metrics[index, FText.HEIGHT] = char_metrics.line_height
        
            
    # ====================================================================================================
    # Align the whole text
    # Before calling this methid, the text must be measured with set_metrics

    def align(self, width=None, align_x='LEFT'):
        
        # ---------------------------------------------------------------------------
        # reset dx and y
        
        self.metrics[:, FText.DX] = 0
        self.metrics[:, FText.Y]  = 0
        
        # ---------------------------------------------------------------------------
        # Loop on the paragraph
        
        y = 0
        for para in self.paras_:
            
            # ---------------------------------------------------------------------------
            # Empty paragraph
            
            if len(para) == 0:
                y -= 1
                continue
            
            # ---------------------------------------------------------------------------
            # Merge words with the next punctuation char
            
            first_word  = para[0]
            words_count = para[1]
            
            ws = self.words_[np.arange(first_word, first_word + words_count)]
            
            merged = np.zeros((0, 3), int)
            w_index = 0
            while w_index <= words_count - 1:
                word = ws[w_index]
                merged = np.append(merged, [word], axis=0)
                
                # ----- A word which is not the last one followed by as single char

                if (word[2] == FText.WORD) and (w_index <= words_count - 2) and (ws[w_index+1, 1] == 1):

                    # ---- It is a quote
                    
                    if Char(self, ws[w_index+1, 0]).char == "'":

                        merged[-1, 1] += 1
                        w_index += 1
                        
                        # Word after if exists
                        if w_index <= words_count - 3:
                            w_index += 1
                            merged[-1, 1] += ws[w_index, 1]
                    
                    # ---- It is a punctuation
                    
                    elif (word[2] == FText.WORD) and (ws[w_index+1, 2] == FText.PUNCT):
                        merged[-1, 1] += 1
                        w_index += 1
                        
                           
                w_index += 1
                
            words = Words(self, merged)
            
            # ---------------------------------------------------------------------------
            # Splitting loop
            
            start_index = 0

            while start_index < len(words):
                
                # ---------------------------------------------------------------------------
                # While we can align words within the width
                
                word = words[start_index]
                word.x = 0
                l = word.width
                last_index = start_index + 1
                
                for i in range(start_index+1, len(words)):
                    
                    next_word = words[i]
                    new_l = l + word.after + word.spaces_after + next_word.width
                    
                    if (width is not None) and (new_l > width):
                        last_index = i
                        break
                    
                    l = new_l
                    last_index = i+1
                    
                # ---------------------------------------------------------------------------
                # Chars location
                
                ch0 = words[start_index].index
                ch1 = words[last_index-1].last_index
                x = 0.
                for i in range(ch0, ch1+1):
                    char = self[i]
                    char.x = x
                    x += char.width + char.after + char.spaces_after
                    char.dx = 0.
                    
                line = Words(self, merged[start_index:last_index])
                line.y = y
                
                # ---------------------------------------------------------------------------
                # Alignment
                
                if width is None:
                    diff = 0.
                else:
                    diff = width - l
                    
                if diff > 0.:
                    
                    # ----- We don't justify the last line
                    
                    align = align_x
                    if (align == 'JUSTIFY') and (last_index == len(words)):
                        align = 'LEFT'
                        
                    # ----- CENTER and RIGHT are simple
        
                    if align == 'CENTER':
                        
                        line.x += diff / 2
                        
                    elif align == 'RIGHT':
                        
                        line.x += diff
                        
                    # ----- JUSTIFY will depend upon we can change the width of the tokens
                        
                    elif align == 'JUSTIFY':
        
                        line.width = width
                                
                # ---------------------------------------------------------------------------
                # Time the create a new line
                
                y -= 1
                start_index = last_index
                
        # ---------------------------------------------------------------------------
        # Build the lines
        
        self.lines_ = np.zeros((-y, 2), int)
        line = 0
        for i in range(len(self.words)):
            w = self.words[i]
            if int(w.y) != -line:
                line = -int(w.y)
                self.lines_[line, 0] = i
                self.lines_[line, 1] = 1
            else:
                self.lines_[line, 1] += 1
                
        # ---------------------------------------------------------------------------
        # Vertcal location
        
        y = 0.
        for i in range(len(self.lines_)):
            line = self.lines[i]
            y -= line.height
            line.y = y
            
    # ====================================================================================================
    # Acces to char indices
    
    @property
    def words_count(self):
        return len(self.words_)
    
    @property
    def lines_count(self):
        return len(self.lines_)
    
    @property
    def paras_count(self):
        return len(self, len(self.paras_))
    
    def word_index(self, word, num=0):
        for i in range(self.words_count):
            if Word(self, self.words_[i]).text == word:
                if num == 0:
                    return i
                num -= 1
        return None
    
    def look_up(self, char=None, word=None, index=None, line_index=None, para_index=None, word_index=None, return_all=False, num=0):

        # ----- Where to look up
        
        if line_index is not None:
            words = self.lines[line_index]
            
        elif para_index is not None:
            words = self.paras[para_index]
            
        elif word_index is not None:
            words = Words(self, [self.words_[word_index]])
            
        else:
            words = self.words
            
        print(words)
        
        # ----- The resulting indices

        indices = []

        # ----- Look for a char
        
        if char is not None:
            for i in words.indices:
                if self.text_[self.chars_[i, 0]] == char:
                    if return_all:
                        indices.append(i)
                    elif num == 0:
                        return [i]
                    else:
                        num -= 1
            return indices
        
        # ----- Look for a word
        
        if word is not None:
            for i in range(len(words)):
                if words[i].text == word:
                    if return_all:
                        indices.extend(words[i].indices)
                    elif num == 0:
                        return words[i].indices
                    else:
                        num -= 1
            return indices
        
        # ----- At a given index
        
        if index is not None:
            return [words.indices[index]]
        
        # ----- All the indices
        
        return words.indices
                
    # ====================================================================================================
    # Plot
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        chars = [self[i].char for i in range(len(self))]
        xywh = np.zeros((len(self), 4), float)
        xywh[:, 0] = self.metrics[:, FText.X]
        xywh[:, 1] = self.metrics[:, FText.Y]
        xywh[:, 2] = self.metrics[:, FText.WIDTH]
        xywh[:, 3] = self.metrics[:, FText.HEIGHT]
        
        for v, s in zip(xywh, chars):
            ax.plot((v[0], v[0] + v[2], v[0] + v[2], v[0], v[0]), (v[1], v[1], v[1]+v[3], v[1]+v[3], v[1]))
            ax.annotate(s, (v[0] + v[2]/2, v[1] + v[3]/2), ha='center', va='center')
            
        plt.show()
        
    # ====================================================================================================
    # A test
    
    @staticmethod
    def Test():
        
        # ----- A test metrics
        
        class Metrics():
            def __init__(self):
                self.space_width = 5
                self.line_height = 1
                
                np.random.seed(0)
                self.widths = np.random.normal(15, 5, (100)).astype(int)
                self.heigths = np.random.uniform(3, 6, 10)
                
            def char_metrics(self, c, char_format):
            
                class Metrics():
                    pass
                
                m = Metrics()
                
                if False:
                    m.xwidth   = 11
                    m.width    = 10
                else:
                    m.xwidth   = self.widths[ord(c) % 100]
                    m.width    = m.xwidth-1
                    
                m.xwidth *= char_format.xscale
                m.width  *= char_format.xscale
                m.after  = m.xwidth - m.width
                
                if True:
                    m.line_height = 1
                else:
                    m.line_height = self.heigths[ord(c)%10]
                    
                m.line_height *= char_format.yscale
                
                
                return m       
            
        # ----- Let's build a text

        text = FText("This is the text, a simple one, to test.\nSeveral paragraphs are written with numbers such as 123.5678 to test split.")
        
        wi = text.word_index("one")
        inds = text.look_up(char="n", word_index=wi)
        word = Word.FromIndices(text, inds)
        word.scale = 2

        char = text.char(13)
        char.scale = 2
        
        text.words[10].yscale = 1.5
        
        text.set_metrics(Metrics())
        
        text.align(width=250, align_x='JUSTIFY')
        text.plot()   
        
        print("word index", text.word_index('Several'))
        
        print(text.look_up(word='to', num=1))

    
#FText.Test()