import sklearn.model_selection
import numpy as np
import random
import string
from nltk import ngrams, FreqDist
import collections
import re


class HangmanSolver(object):
    def __init__(self, weight_length = 4.2, weight_count = 0.3, correct_word = "", training_dictionary_location = "words_250000_train.txt", test_dictionary_location = "", train = True, test_size = 0.33):
        self.weight_length = weight_length
        self.weight_count = weight_count
        self.train = train

        self.guessed_letters = []
        if correct_word != "" :
            self.givenExternalWord = True
        else:
            self.givenExternalWord = False
        
        self.word = ""
        self.correct_word = correct_word

        self.training_dictionary, self.test_dictionary = self.build_dictionaries(training_dictionary_location, test_dictionary_location, test_size)

        self.allNGrams = self.generateNGrams(self.training_dictionary, 5)
        self.current_dictionary = []
        self.alphabet = list(string.ascii_lowercase)
        self.vowels = list(["a", "e", "i", "o", "u"])
        self.guess_list = []
        self.has_won = False
        self.gameLog = []
        #self.init_grams()
        
    def build_dictionaries(self, train_loc, test_loc, test_size):
        if(self.train == True):
            #If we are in training mode, let's assume that we have one dictionary we are dividing in a test/train split.
            #if we are in training mode, it would make sense that this is training_dictionary_location...
            full_dict = self.build_dictionary(train_loc)
            dict_train, dict_test = sklearn.model_selection.train_test_split(full_dict, test_size = test_size) #test_size is adjustable, it doesn't do anything if not training though
        else:
            #else load both dicts directly
            dict_train = self.build_dictionary(train_loc)   
            dict_test = self.build_dictionary(test_loc)

        return [dict_train, dict_test]

    def change_weights(self, wl, wc):
        self.weight_length = wl
        #self.weight_probabolity = wp
        self.weight_count = wc
        #can probably mess with more weights...


    def generateNGrams(self, vocab, n):
        #return the ngrams as dictionaries
        #I used the freq_dists and ngrams from nltk but they are way too slow
        #this is a little confusing because I am loading words from a dictionary (not the data type) of words...
        #let's call it vocab here instead
        allNGrams = []
        for k in range(2, n+1):
            ngrams = {}
            for word in vocab:
                word = "." + word + "." # pad with start and end bits
                if len(word) < k:
                    continue
                for i in range(len(word) - k + 1):
                    gram = word[i:i+k]
                    if gram not in ngrams:
                        ngrams[gram] = 1
                    else:
                        ngrams[gram] += 1
            allNGrams.append(ngrams.copy())
        return allNGrams
        
        
    def generate_correct_word(self):
        #Ok, so by default the code will generate a random correct word from the full dictionary
        #You can also initialize the class with an external correct word, or set one after initialization
        if self.givenExternalWord == False:
            self.correct_word = random.choice(self.test_dictionary)
    
    def set_correct_word(self, givenWord):
        self.correct_word = givenWord
        self.givenExternalWord = True

    def reset_correct_word(self):
        #this is just to reset the given correct word in case that is necessary
        self.correct_word = ""
        self.givenExternalWord = False


    def find_potential_guesses(self, allGrams, input_word, guess_list):
        
        potential_guesses = []
        remove_set = set(guess_list)
        remove_set.add(None)

        substr_list = []
        input_word = "." + input_word + "." #pad beginning and end with something to indicate beginning/end

        for i,char in enumerate(input_word):
            if char == "_": #found a blank
                start, end = i, i
                #print(i)
                #get longest substring containing the blank
                while start > 0 and input_word[start-1] != "_":
                    start-=1
                    #print(start)
                while end < len(input_word) - 1 and input_word[end+1] != "_":
                    end += 1
                    #print(end)
                substring = input_word[start : end + 1]
                substr_list.append(substring)
        #print(substr_list)

        for substr in substr_list:
            max_dim = min(len(substr), 5)
            current_dim = max_dim
            match_temp_array = []

            while current_dim >= 2:
                ngramDict = allGrams[current_dim - 2]

                grams = ngrams(substr, current_dim)
                subsubstr_list = ["".join(g) for g in grams if g.count("_") == 1]
                subsubstr_list = [x for x in subsubstr_list if (x != "_." and x != "._")]
                #print(subsubstr_list)
                
                for subsubstr in subsubstr_list:
                    #print(subsubstr)
                    #print(ngramDict)
                    idx = subsubstr.find("_")
                    total_match_count = 0
                    for l in string.ascii_lowercase:
                        filledSubstr = subsubstr[0:idx] + l + subsubstr[idx+1::]
                        #print(filledSubstr)
                        if(ngramDict.get(filledSubstr)):
                            count = ngramDict[filledSubstr]
                            match_temp_array.append([l, filledSubstr, count])
                            total_match_count += count
                            #print(l, count, filledSubstr, sep = " ")
                    for item in match_temp_array:
                        if (item[0] not in guess_list and total_match_count > 0):
                            potential_guesses.append([item[0], current_dim, item[2] / total_match_count, total_match_count])
                
                #print(match_temp_array)
                if(len(match_temp_array) > 0):
                    #print(match_temp_array)
                    #print(potential_guesses)
                    break

                current_dim -= 1
        return potential_guesses
        

    def evaluate_potential_guesses(self, potential_guesses):
        #print(potential_guesses)
        if(potential_guesses == []):
            return "!" # no n-gram fits
        # Initialize dictionary to keep scores for each guess
        scores = {}

        # Weighting factors, these are the main thing to tweak...
        weight_length = self.weight_length  # Giving more weight to the length of the n-gram
        weight_count = self.weight_count

        for guess, length, probability, tot_count in potential_guesses:
            guess_letter = list(guess)[0]  # Extract the letter from the set
            #if length == 2:
            #    continue

            # Calculate the score for this guess
            #score = probability * weight_probability + (length * weight_length) + weight_count * np.log(tot_count)
            score = np.log(probability * tot_count) * weight_count + weight_length * length

            # Add or update the score for this letter in the scores dictionary
            if guess_letter in scores:
                scores[guess_letter] += score
            else:
                scores[guess_letter] = score

        # Determine the best guess by finding the maximum score
        #print(scores)
        if len(scores) == 0:
            return "!"
        best_guess = max(scores, key=scores.get)
        return best_guess



    def guess(self, word): # word input example: "_ p p _ e "

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word.replace(" ", "") #I am putting this here in case I forget later to remove the space stuff when I paste this into the solution
        #clean_word = word.replace("_",".") #I am using underscores

        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        #print(current_dictionary[0])
        
        # iterate through all of the words in the old plausible dictionary
        reMatchWord = clean_word.replace("_",".")
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(reMatchWord, dict_word):
                new_dictionary.append(dict_word)


        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()
        #print(sorted_letter_count)                
        
        #guess_letter = '!'

        all_guesses = self.find_potential_guesses(self.allNGrams, word, self.guessed_letters)
        #print(all_guesses)
        ngram_guess = self.evaluate_potential_guesses(all_guesses)
        
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        #print(sorted_letter_count)
        most_common_guess = "!"

        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                most_common_guess = letter
                #print(most_common_guess)
                break
        
        if(ngram_guess != "!"):
            return ngram_guess
        if(most_common_guess != "!"):
            return most_common_guess
        
        #ok, so we will only get down here if the current dictionary is EMPTY and there are no ngram matches...
        #what do we do in this case... well let's filter based on the letters
        #It is rare that this happens...
        backup_dictionary = self.training_dictionary

        last_dictionary = backup_dictionary.copy()

        incorrect_set = set(self.guessed_letters) - set(word)
        #correct_set = set(self.guessed_letters) - incorrect_set

        backup_dictionary = [word for word in backup_dictionary if set(word).isdisjoint(incorrect_set)]
        backup_dict_string = "".join(backup_dictionary)
        bc = collections.Counter(backup_dict_string)
        backup_letter_count = bc.most_common()

        backup_guess = "!"
        for letter, instance_count in backup_letter_count:
            if letter not in self.guessed_letters:
                backup_guess = letter
                #print(most_common_guess)
                break
        if backup_guess != "!":
            return backup_guess
        
        #Ok... but what if this also fails?
        #well... let's just default back to the order of the entire dictionary
        #I guess in some freak accident the code could be given a training dictionary where there were only 5 letters in all of the words and the test word has one that isn't there... but that would be pretty insane.
        last_dict_string = "".join(last_dictionary)
        lc = collections.Counter(last_dict_string)
        last_letter_count = lc.most_common()

        for letter, instance_count in last_letter_count:
            if letter not in self.guessed_letters:
                last_guess = letter
                #print(most_common_guess)
                break
        
        return last_guess


    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def handle_guess(self, guessLetter, currentWord, ans):
        #I have to convert the strings to lists, find indices of the letter (if present) and then smush the lists back into strings and spit it out
        indices = [i for i, x in enumerate(list(ans)) if x == guessLetter]
        wordList = list(currentWord)
        for i in indices:
            wordList[i] = guessLetter #now I replace the occurrences of the letter
        word = "".join(wordList)
        return word

    def get_winstate(self):
        return self.has_won
    
    def get_gamelog(self):
        return self.gameLog
        
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.gameLog = []
        self.guessed_letters = []
        self.current_dictionary = self.training_dictionary
        self.has_won = False
        self.generate_correct_word()
        word = "_" * len(self.correct_word) #I should make this a member of self right?

        tries_remains = 6 #I think this needs to just be a magic number?
        
        self.gameLog.append([self.correct_word, word, self.guessed_letters.copy(), tries_remains])
        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}. Word: {1}.".format(tries_remains, word))
        while tries_remains >0:
            #tries_remains -=1 tries remains decreases after WRONG answer

            #get guessed letter
            guess_letter = self.guess(word)

            #append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)
            

            if verbose:
                print("Guessing letter: {0}".format(guess_letter))

            #apply guessed letter to the word
            new_word = self.handle_guess(guess_letter, word, self.correct_word) 

            #check if we have the word
            if new_word == self.correct_word:
                word = new_word
                if verbose:
                    print("Success! the word was: %s" % word)
                self.has_won = True
                self.gameLog.append([self.correct_word, word, self.guessed_letters.copy(), tries_remains])
                break
            
            if new_word == word:
                #this means that we did not get a correct guess
                #decrease number of tries
                if verbose:
                    print("Incorrect guess {0}, # of tries remaining: {1}. Word: {2}.".format(guess_letter,tries_remains, word))
                tries_remains -=1
            else:
                #We have a correct letter guessed, but not the complete word
                #don't decrement tries_remains
                word = new_word
                if verbose:
                    print("Got a Letter, {0}, # of tries remaining: {1}. Word: {2}.".format(guess_letter,tries_remains, word))

            self.gameLog.append([self.correct_word, word, self.guessed_letters.copy(), tries_remains])
        if tries_remains == 0:
            if verbose:
                print("You Lose, the answer was: %s" % self.correct_word)
                

