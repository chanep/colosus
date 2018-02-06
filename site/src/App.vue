<template>
  <div class="container">
    <div class="board"><Board/></div>
    <div class="commands"><CommandPanel/></div>
  </div>
</template>

<script>
  import Board from "./components/Board.vue"
  import CommandPanel from "./components/CommandPanel.vue"

  export default {
    name: 'app',
    components: {Board, CommandPanel}
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
    .commands{
      /* margin-left: 10vh */
    }
</style>
