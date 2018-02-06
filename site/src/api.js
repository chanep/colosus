import socketio from 'socket.io-client';
import { log } from 'util';

let socket = socketio('http://localhost:5003');

socket.on('connect', () => {
    console.log("conectado!!")
})

socket.on('status_update', (status) => {
    console.log("stat_update!")
    console.log(status)
})

socket.on('disconnect', () => {
    console.log("disconnected!!")
  });

export default {
    newGame(blackHuman, whiteHuman, iterations){
        socket.emit('new_game', {blackHuman: blackHuman, whiteHuman:whiteHuman, iterations: iterations});
    },
    move(rank, file){
        socket.emit('move', {rank: rank, file: file});
    }
};