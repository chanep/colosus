<template>
  <div class="container">
      <div class="left">
        <div class="text-group">
            <label for="iterations">Iterations</label>
            <input name="iterations" v-model.number="iterations" type="number">
        </div>
        <div class="text-group">
            <div class="radio-group">
                <input type="radio" v-model="humanSide" :value="humanSideValues.black">
                <label>Black</label>
            </div>
            <div class="radio-group">
                <input type="radio" v-model="humanSide" :value="humanSideValues.white">
                <label>White</label>
            </div>
            <div class="radio-group">
                <input type="radio" v-model="humanSide" :value="humanSideValues.both">
                <label>Human vs. Human</label>
            </div>
            <div class="radio-group">
                <input type="radio" v-model="humanSide" :value="humanSideValues.none">
                <label>Colosus vs. Colosus</label>
            </div>
        </div>
      </div>
      <div class="right">
        <div class="group">
            <button @click="newGame">
                New Game
            </button>
        </div>
      </div>
  </div>
</template>

<script>
    import store from "../store";
    import api from "../api";

    export default {
        data() {
            return {
                iterations: 256,
                humanSide: "black",
                humanSideValues: {
                    black: "black",
                    white: "white",
                    both: "both",
                    none: "none"
                }
            };
        },
        methods: {
            newGame(){
                let blackHuman = this.humanSide == "black" || this.humanSide == "both";
                let whiteHuman = this.humanSide == "white" || this.humanSide == "both";
                api.newGame(blackHuman, whiteHuman, this.iterations);
            }
        }
    };
</script>

<style scoped>
    .container{
        display: flex;
        flex-direction: row;
        padding: 20px;
    }
    .text-group{
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }
    .radio-group{
        display: flex;
        flex-direction: row;
        
    }
    .left{
        display: flex;
        flex-direction: column;
        margin-right: 20px;
    }
    input[type="radio"]{
        margin-right: 5px;
    }
</style>