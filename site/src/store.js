let gameStatus = {
    board: null,
    humanTurn: false,
    lastMove: null,
    winner: null,
    value: null,
    depth: null,
    error: null,
    sideToMove: null,
    inProgress: false,
    winLine: null
}

// let gameSettings = {
//     human
// }

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

gameStatus.board = board;

let store = {
    gameStatus: gameStatus,
    appState : appState,
    updateStatus(status){
        this.gameStatus.board = status.board;
        this.gameStatus.humanTurn = status.humanTurn;
        this.gameStatus.lastMove = status.lastMove;
        this.gameStatus.winner = status.winner;
        this.gameStatus.value = status.value;
        this.gameStatus.depth = status.depth;
        this.gameStatus.nodes = status.nodes;
        this.gameStatus.error = status.error;
        this.gameStatus.sideToMove = status.sideToMove;
        this.gameStatus.inProgress = status.inProgress;
        this.gameStatus.winLine = status.winLine;
        
    }
}

export default store;