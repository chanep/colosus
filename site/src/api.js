import socketio from 'socket.io-client';
import store from './store';

let socket = socketio('http://localhost:5003', {transports: [/*'websockets',*/ 'polling']});

socket.on('connect', () => {
    console.log('conectado!!');
});

socket.on('status_update', (status) => {
    console.log(status);
    store.updateStatus(status);
});

socket.on('disconnect', () => {
    console.log('disconnected!!');
});

export default {
    newGame(blackHuman, whiteHuman, iterations){
        socket.emit('new_game', {blackHuman: blackHuman, whiteHuman:whiteHuman, iterations: iterations});
    },
    move(rank, file){
        socket.emit('move', {rank: rank, file: file});
    }
};