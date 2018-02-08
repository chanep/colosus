<template>
  <div class="container">
    <div class="board"><Board/></div>
    <div class="left">
      <div class="panels">
        <PlayerPanel class="player-panel" side=0 />
        <PlayerPanel class="player-panel" side=1 />
      </div>
      <Notification class="notification"/>
      <CommandPanel/>
    </div>
  </div>
</template>

<script>
  import Board from "./components/Board.vue"
  import CommandPanel from "./components/CommandPanel.vue"
  import PlayerPanel from "./components/PlayerPanel.vue"
  import Notification from "./components/Notification.vue"

  export default {
    name: 'app',
    components: {Board, CommandPanel, PlayerPanel, Notification}
  }
</script>

<style scoped>
    .container{
      display: flex;
      flex-direction: row;
      margin-top: 20px;
    }
    .board{
      width: max-content;
    }
    .left{
      display: flex;
      flex-direction: column;
    }
    .notification{
      flex-grow: 1;
    }
    .panels{
      display: flex;
      flex-direction: row;
    }
    .player-panel{
      margin: 15px;
    }
</style>
