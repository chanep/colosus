import socketio from 'socket.io-client';
import { log } from 'util';

console.log('api load!');

let socket = socketio('http://localhost:5003');

socket.on('connect', () => {
    console.log("conectado!!")
    socket.emit('new_game', {blackHuman: true, whiteHuman:true, iterations: 256});
    console.log('emited!');
})

socket.on('status_update', (status) => {
    console.log("stat_update!")
    console.log(status)
})

socket.on('disconnect', () => {
    console.log("disconnected!!")
  });

export default {};