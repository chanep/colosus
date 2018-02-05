<template>
  <table class="board">
      <tr v-for="(rank, rindex) in board">
          <td class="square" v-for="square in rank">
              <img class="stone" src="../assets/stoneW.svg" alt="">
          </td>
          <td class="coord">{{15 - rindex}}</td>
      </tr>
      <tr>
          <td class="coord" v-for="file in 15">{{file}}</td>
      </tr>
  </table>
</template>

<script>
import store from "../store"
export default {
  data () {
    return {
        board: store.state.board.slice().reverse()
        }
    }
}
</script>

<style scoped>
    .board {
        border-spacing: 0;
        border-collapse: collapse;
    }
    .square {
        margin: 0;  
        border: 1px solid;
        width: 20px;
        height: 20px;
    }
    .stone{
        display: block;
        width: 100%;
        height: 100%;
    }
</style>


