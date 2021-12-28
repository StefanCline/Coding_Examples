# Tic Tac Toe Game
print('')
# Preconditions

player_1 = input("Input player 1's name: ")
player_2 = input("Input player 2's name: ")
game_count = 0
player_choice = player_1
move = ''
selection = ''
play_again_check = 0
keep_playing = True
keep_playing_game = True
valid_inputs = [1,2,3,4,5,6,7,8,9]
move_count = 0

#Game Board

row1 = [" ","|"," ","|"," "]
row2 = ["-","-","-","-","-"]
row3 = [" ","|"," ","|"," "]
row4 = ["-","-","-","-","-"]
row5 = [" ","|"," ","|"," "]

# Game Functions

def player1_move(choice):
    if choice in range(1,4):
        if move == 1:
            row5[0] = 'X'
        if move == 2:
            row5[2] = 'X'
        if move == 3:
            row5[4] = 'X'
    if choice in range(4,7):
        if move == 4:
            row3[0] = 'X'
        if move == 5:
            row3[2] = 'X'
        if move == 6:
            row3[4] = 'X'
    if choice in range(7,10):
        if move == 7:
            row1[0] = 'X'
        if move == 8:
            row1[2] = 'X'
        if move == 9:
            row1[4] = 'X'
                        
def player2_move(choice):
    if choice in range(1,4):
        if move == 1:
            row5[0] = 'O'
        if move == 2:
            row5[2] = 'O'
        if move == 3:
            row5[4] = 'O'
    if choice in range(4,7):
        if move == 4:
            row3[0] = 'O'
        if move == 5:
            row3[2] = 'O'
        if move == 6:
            row3[4] = 'O'
    if choice in range(7,10):
        if move == 7:
            row1[0] = 'O'
        if move == 8:
            row1[2] = 'O'
        if move == 9:
            row1[4] = 'O'

def gameboard_vis(r1,r2,r3,r4,r5):
    r_1 = (str(r1).replace("'",''))
    print(r_1.replace(',',''))
    r_2 = (str(r2).replace("'",''))
    print(r_2.replace(',',''))
    r_3 = (str(r3).replace("'",''))
    print(r_3.replace(',',''))
    r_4 = (str(r4).replace("'",''))
    print(r_4.replace(',',''))
    r_5 = (str(r5).replace("'",''))
    print(r_5.replace(',',''))

# Overall Game Loop

while keep_playing == True:

    # Checks if the players want to play again after completing at least 1 game

    if game_count > 0: 
        selection = input("Do you want to play again? (Y/N): ")
        while play_again_check == 0:
            if selection == 'Y':
                keep_playing = True
                keep_playing_game = True
                valid_inputs = [1,2,3,4,5,6,7,8,9]
                row1 = [" ","|"," ","|"," "]
                row2 = ["-","-","-","-","-"]
                row3 = [" ","|"," ","|"," "]
                row4 = ["-","-","-","-","-"]
                row5 = [" ","|"," ","|"," "]
                play_again_check = 1
            elif selection == 'N':
                keep_playing = False
                play_again_check = 1
            else:
                print("Please enter either Y or N.")

    # Start of the loop that plays the game

    play_again_check = 0

    while keep_playing_game == True and keep_playing == True:
        
        # Player making their move
        acceptable_move = False
        while acceptable_move == False:    
            move = (input("{} please make your move (1-9). ".format(player_choice)))
            if move.isdigit() == False:
                print("Please enter a number 1-9.")
                acceptable_move = False
            elif int(move) in valid_inputs:
                acceptable_move = True
                move = int(move)
                if player_choice == player_1:
                    player1_move(move)
                    gameboard_vis(row1,row2,row3,row4,row5)
                    player_choice = player_2
                    valid_inputs.remove(move)
                else: 
                    player2_move(move)
                    gameboard_vis(row1,row2,row3,row4,row5)
                    player_choice = player_1
                    valid_inputs.remove(move)
            else:                 
                acceptable_move = False

        #Horizontal Wins
        if row5[0] == 'X' and row5[2] == 'X' and row5[4] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row3[0] == 'X' and row3[2] == 'X' and row3[4] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row1[0] == 'X' and row1[2] == 'X' and row1[4] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row5[0] == 'O' and row5[2] == 'O' and row5[4] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        if row3[0] == 'O' and row3[2] == 'O' and row3[4] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        if row1[0] == 'O' and row1[2] == 'O' and row1[4] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        
        #Vertical Wins
        if row5[0] == 'X' and row3[0] == 'X' and row1[0] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row5[2] == 'X' and row3[2] == 'X' and row1[2] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row5[4] == 'X' and row3[4] == 'X' and row1[4] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row5[0] == 'O' and row3[0] == 'O' and row1[0] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        if row5[2] == 'O' and row3[2] == 'O' and row1[2] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        if row5[4] == 'O' and row3[4] == 'O' and row1[4] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        
        #Diagonal Wins
        if row5[0] == 'X' and row3[2] == 'X' and row1[4] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row3[4] == 'X' and row3[2] == 'X' and row3[0] == 'X':
            print("Congratulations {}, you won!".format(player_1))
            game_count += 1
            keep_playing_game = False
        if row5[0] == 'O' and row3[2] == 'O' and row1[4] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        if row3[4] == 'O' and row3[2] == 'O' and row3[0] == 'O':
            print("Congratulations {}, you won!".format(player_2))
            game_count += 1
            keep_playing_game = False
        
        #Stalemate
        if valid_inputs == []:
            game_count += 1 
            print("This game is a stalemate. At least no one lost!")
            keep_playing_game = False
print("Game Over.")