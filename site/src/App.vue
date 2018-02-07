<template>
  <div class="container">
    <div class="board"><Board/></div>
    <div class="left">
      <CommandPanel/>
      <PlayerPanel side=0 />
    </div>
  </div>
</template>

<script>
  import Board from "./components/Board.vue"
  import CommandPanel from "./components/CommandPanel.vue"
  import PlayerPanel from "./components/PlayerPanel.vue"

  export default {
    name: 'app',
    components: {Board, CommandPanel, PlayerPanel}
  }
</script>

<style scoped>
    .container{
      display: flex;
      flex-direction: row;
    }
    .board{
      width: max-content;
    }
    .left{
      display: flex;
      flex-direction: column;
    }
</style>
