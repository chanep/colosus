let state = {
    board: null,
    humanTurn: false,
    lastMove: null,
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

let appState = {
    viewColosusValue: false
}

state.board = board;

let store = {
    state: state,
    appState : appState,
    updateStatus(status){
        this.state = status
    }
}

export default store;