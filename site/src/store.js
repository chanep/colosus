let state = {
    board: null,
    human_turn: false,
    last_move: null,
    winner: null,
    value: null,
    error: null
}

let board = [];
for(let r=0;r<15;r++){
    let rank = []
    for(let f=0;f<15;f++){
        rank.push('-')
    }
    board.push(rank)
}

board[0][0] = "X";

state.board = board;

let store = {
    state: state
}

export default store;